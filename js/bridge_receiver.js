/**
 * BridgeReceiver front-end extension.
 *
 * The browser userscript sends images to the Python bridge server. This file
 * keeps the current ComfyUI graph synced to that server so "send and run" can
 * queue the workflow from Python even when the ComfyUI tab is in the background.
 */

import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";
import { installLlmProviderUi, refreshLlmProviderLabel } from "./llm_provider_manager.js";

const BRIDGE_URL = "http://127.0.0.1:18901";
const POLL_INTERVAL = 2000;
const SYNC_INTERVAL = 1500;

function widgetByName(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function modelLooksThinkingCapable(model) {
    const text = String(model || "").toLowerCase();
    return [
        "qwen3",
        "qwq",
        "deepseek-r1",
        "deepseek-reasoner",
        "reasoner",
        "reasoning",
        "thinking",
        "o1",
        "o3",
        "o4",
    ].some((hint) => text.includes(hint));
}

function setThinkingModelDetails(node, details) {
    node.aerinceThinkingModels = {};
    for (const item of details || []) {
        if (item?.id) {
            node.aerinceThinkingModels[item.id] = !!item.thinking_capable;
        }
    }
}

function refreshThinkingWidgetLabel(node) {
    const model = widgetByName(node, "model")?.value || "";
    const thinkingWidget = widgetByName(node, "thinking_strength");
    if (!thinkingWidget) return;

    const detail = node.aerinceThinkingModels?.[model];
    const capable = detail ?? modelLooksThinkingCapable(model);
    thinkingWidget.label = capable ? "thinking_strength (detected)" : "thinking_strength";
}

function setModelOptions(widget, models) {
    if (!widget || !Array.isArray(models)) return;
    const values = models.length ? models : ["refresh_models_first"];
    widget.options = widget.options || {};
    widget.options.values = values;
    if (!values.includes(widget.value)) {
        widget.value = values[0];
        widget.callback?.(widget.value);
    }
}

function replaceModelTextWidgetWithCombo(node) {
    const existing = widgetByName(node, "model");
    const currentValue = existing?.value || "refresh_models_first";
    if (existing?.type === "combo") {
        existing.options = existing.options || {};
        existing.options.values = existing.options.values?.length ? existing.options.values : [currentValue];
        return existing;
    }

    if (existing && Array.isArray(node.widgets)) {
        const index = node.widgets.indexOf(existing);
        if (index !== -1) {
            node.widgets.splice(index, 1);
        }
    }

    const values = currentValue && currentValue !== "" ? [currentValue] : ["refresh_models_first"];
    const combo = node.addWidget("combo", "model", values[0], () => {
        refreshThinkingWidgetLabel(node);
        node.setDirtyCanvas?.(true, true);
    }, { values });

    if (existing && Array.isArray(node.widgets)) {
        const addedIndex = node.widgets.indexOf(combo);
        const targetIndex = Math.min(3, node.widgets.length - 1);
        if (addedIndex !== -1 && addedIndex !== targetIndex) {
            node.widgets.splice(addedIndex, 1);
            node.widgets.splice(targetIndex, 0, combo);
        }
    }

    return combo;
}


function setWidgetValue(widget, value) {
    if (!widget) return;
    widget.value = value;
    if (widget.inputEl) widget.inputEl.value = value;
    if (widget.element) widget.element.value = value;
    widget.callback?.(value);
}

function getWidgetNumber(node, name, fallback) {
    const value = widgetByName(node, name)?.value;
    const num = Number(value);
    return Number.isFinite(num) ? num : fallback;
}

function migrateLlmWidgetValues(node, info) {
    const values = info?.widgets_values;
    if (!Array.isArray(values) || widgetByName(node, "max_tokens")) return info;
    const timeoutIndex = node.widgets?.findIndex((widget) => widget.name === "timeout");
    const thinkingIndex = node.widgets?.findIndex((widget) => widget.name === "thinking_strength");
    // Buttons may be saved or omitted; length alone cannot identify the old layout.
    // Use the live order, including the existing model text-to-combo reorder.
    if (timeoutIndex < 0 || thinkingIndex !== timeoutIndex + 1 ||
        !Number.isFinite(values[timeoutIndex]) ||
        !Number.isFinite(values[timeoutIndex + 1]) ||
        typeof values[thinkingIndex + 1] !== "string") return info;
    const migrated = values.slice();
    migrated.splice(timeoutIndex, 1);
    return { ...info, widgets_values: migrated };
}

async function resolveConnectedImageDataUrl(node) {
    const imageInput = node.inputs?.find((input) => input.name === "image");
    if (!imageInput || imageInput.link == null) {
        throw new Error("请先连接 image 输入");
    }

    const link = app.graph.links?.[imageInput.link];
    if (!link) {
        throw new Error("找不到 image 连接");
    }

    const sourceNode = app.graph.getNodeById(link.origin_id);
    if (!sourceNode) {
        throw new Error("找不到上游图片节点");
    }

    // Prefer source node preview / imgs first.
    const candidates = [];
    if (Array.isArray(sourceNode.imgs)) candidates.push(...sourceNode.imgs);
    if (sourceNode.imageIndex != null && sourceNode.imgs?.[sourceNode.imageIndex]) {
        candidates.unshift(sourceNode.imgs[sourceNode.imageIndex]);
    }
    if (sourceNode.image) candidates.push(sourceNode.image);

    for (const img of candidates) {
        if (!img) continue;
        if (typeof img === "string") {
            if (img.startsWith("data:")) return img;
            // Try fetch absolute/relative url
            try {
                const resp = await fetch(img);
                const blob = await resp.blob();
                return await blobToDataUrl(blob);
            } catch (e) {
                // continue
            }
        }
        if (img?.src) {
            if (img.src.startsWith("data:")) return img.src;
            try {
                const resp = await fetch(img.src);
                const blob = await resp.blob();
                return await blobToDataUrl(blob);
            } catch (e) {
                // canvas fallback below
            }
            if (img.complete && img.naturalWidth) {
                return imageElementToDataUrl(img);
            }
        }
    }

    // Load Image widget fallback (LoadImage style nodes)
    const imageWidget = widgetByName(sourceNode, "image");
    if (imageWidget?.value) {
        const name = String(imageWidget.value);
        const urls = [
            `/view?filename=${encodeURIComponent(name)}&type=input`,
            `/view?filename=${encodeURIComponent(name)}&type=temp`,
            `/view?filename=${encodeURIComponent(name)}&type=output`,
        ];
        for (const url of urls) {
            try {
                const resp = await fetch(url);
                if (!resp.ok) continue;
                const blob = await resp.blob();
                if (blob.size > 0) return await blobToDataUrl(blob);
            } catch (e) {
                // try next
            }
        }
    }

    throw new Error("无法读取上游图片。请先让图片节点生成预览，或使用 Load Image 节点。");
}

function blobToDataUrl(blob) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
    });
}

function imageElementToDataUrl(img) {
    const canvas = document.createElement("canvas");
    canvas.width = img.naturalWidth || img.width;
    canvas.height = img.naturalHeight || img.height;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(img, 0, 0);
    return canvas.toDataURL("image/png");
}

async function reverseImagePromptEditable(node) {
    if (node.aerinceReverseInFlight) return;
    node.aerinceReverseInFlight = true;
    const button = widgetByName(node, "reverse_now");
    const oldLabel = button?.name;

    try {
        if (button) button.name = "反推中...";
        node.setDirtyCanvas?.(true, true);
        const imageDataUrl = await resolveConnectedImageDataUrl(node);
        const payload = {
            image_data_url: imageDataUrl,
            base_url: widgetByName(node, "base_url")?.value || "",
            api_key: widgetByName(node, "api_key")?.value || "",
            model: widgetByName(node, "model")?.value || "",
            system_prompt: widgetByName(node, "system_prompt")?.value || "",
            user_prompt: widgetByName(node, "user_prompt")?.value || "",
            temperature: getWidgetNumber(node, "temperature", 0.2),
            timeout: getWidgetNumber(node, "timeout", 300),
            thinking_strength: widgetByName(node, "thinking_strength")?.value || "auto",
        };

        const response = await api.fetchApi("/aerince/llm/reverse-image", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
        });
        const data = await response.json();
        if (!response.ok || !data?.success) {
            throw new Error(data?.error || `HTTP ${response.status}`);
        }
        if (typeof data.prompt !== "string" || !data.prompt.trim()) {
            throw new Error("模型未返回有效的提示词");
        }

        setWidgetValue(widgetByName(node, "prompt_text"), data.prompt);
        if (typeof data.image_hash === "string" && data.image_hash.trim()) {
            setWidgetValue(widgetByName(node, "source_image_hash"), data.image_hash);
        }
        node.setDirtyCanvas?.(true, true);
        app.graph?.setDirtyCanvas?.(true, true);
        console.log("[AerinceTools] reverse-image filled prompt_text");
    } catch (error) {
        console.error("[AerinceTools] reverse-image failed:", error);
        alert(`Aerince Tools 反推失败\n${error.message || error}`);
    } finally {
        node.aerinceReverseInFlight = false;
        if (button) button.name = oldLabel || "反推";
        node.setDirtyCanvas?.(true, true);
    }
}

async function refreshLlmModels(node, throwOnError = false) {
    const generation = node.aerinceModelsGeneration = (node.aerinceModelsGeneration || 0) + 1;
    const baseUrl = widgetByName(node, "base_url")?.value || "";
    const apiKey = widgetByName(node, "api_key")?.value || "";
    const modelWidget = widgetByName(node, "model");
    const buttonWidget = node.aerinceModelRefreshButton || widgetByName(node, "refresh_models");
    const current = () => generation === node.aerinceModelsGeneration &&
        baseUrl === (widgetByName(node, "base_url")?.value || "") &&
        apiKey === (widgetByName(node, "api_key")?.value || "");

    if (buttonWidget) buttonWidget.name = "refreshing...";
    node.setDirtyCanvas?.(true, true);

    try {
        const response = await api.fetchApi("/aerince/llm/models", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ base_url: baseUrl, api_key: apiKey }),
        });
        const data = await response.json();
        if (!current()) return;
        if (!response.ok || !data?.success) {
            throw new Error(data?.error || `HTTP ${response.status}`);
        }
        setModelOptions(modelWidget, data.models || []);
        setThinkingModelDetails(node, data.model_details || []);
        refreshThinkingWidgetLabel(node);
        console.log(`[AerinceTools] Loaded ${data.models?.length || 0} LLM models`);
    } catch (error) {
        if (!current()) return;
        console.error("[AerinceTools] Failed to refresh LLM models:", error);
        if (throwOnError) throw error;
        alert(`Aerince Tools: failed to refresh models\n${error.message || error}`);
    } finally {
        if (buttonWidget && generation === node.aerinceModelsGeneration) buttonWidget.name = "refresh_models";
        node.setDirtyCanvas?.(true, true);
    }
}

const ext = {
    name: "BridgeReceiver",

    async setup() {
        let lastFallbackQueuedUid = 0;
        let lastSyncText = "";
        let syncing = false;

        async function syncWorkflow(force = false) {
            if (syncing || !app.graphToPrompt) return;
            syncing = true;
            try {
                const graphData = await app.graphToPrompt();
                const prompt = graphData?.output;
                if (!prompt || Object.keys(prompt).length === 0) return;

                const payload = {
                    prompt,
                    workflow: graphData.workflow,
                    client_id: api.clientId,
                };
                const text = JSON.stringify(payload);
                if (!force && text === lastSyncText) return;

                const resp = await fetch(BRIDGE_URL + "/bridge/workflow", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: text,
                });
                if (resp.ok) {
                    lastSyncText = text;
                }
            } catch (e) {
                // The Python bridge may not be loaded yet; keep trying quietly.
            } finally {
                syncing = false;
            }
        }

        async function pollStatus() {
            try {
                await syncWorkflow(false);

                const resp = await fetch(BRIDGE_URL + "/bridge/status");
                if (!resp.ok) return;
                const status = await resp.json();

                const triggerUid = status.last_trigger_uid || 0;
                const directQueuedUid = status.last_direct_queued_uid || 0;

                if (
                    triggerUid > 0 &&
                    triggerUid !== directQueuedUid &&
                    triggerUid !== lastFallbackQueuedUid
                ) {
                    lastFallbackQueuedUid = triggerUid;
                    scheduleQueue(triggerUid, 100);
                }
            } catch (e) {
                // Background tabs can throttle timers; the Python direct queue path
                // is now the primary trigger, so polling is only a fallback.
            }
        }

        function scheduleQueue(uid, delay) {
            setTimeout(async () => {
                try {
                    await syncWorkflow(true);
                    await app.queuePrompt(-1, 1);
                    console.log(`[BridgeReceiver] Fallback queued prompt  uid=${uid}`);
                } catch (e) {
                    console.error(`[BridgeReceiver] Fallback queue failed  uid=${uid}:`, e);
                }
            }, delay);
        }

        api.addEventListener("bridge-trigger", (event) => {
            const { uid } = event.detail || {};
            if (!uid || uid === lastFallbackQueuedUid) return;
            lastFallbackQueuedUid = uid;
            scheduleQueue(uid, 200);
        });

        syncWorkflow(true);
        setInterval(() => syncWorkflow(false), SYNC_INTERVAL);
        setInterval(pollStatus, POLL_INTERVAL);

        console.log("[BridgeReceiver] Ready: workflow sync + direct backend queue fallback active");
    },

    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "BridgeReceiver" || nodeData.name === "BridgeImageLoader") {
            const onNodeCreated = nodeType.prototype.onNodeCreated;

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);

                const imageWidget = this.widgets?.find((w) => w.name === "image");
                if (imageWidget) {
                    imageWidget.options = imageWidget.options || {};
                    imageWidget.options.image_upload = true;
                }

                return r;
            };
        }

        if (
            nodeData.name === "LLMImagePromptReverse" ||
            nodeData.name === "LLMImagePromptReverseEditable" ||
            nodeData.name === "LLMPromptOptimizer"
        ) {
            const onNodeCreated = nodeType.prototype.onNodeCreated;
            const configure = nodeType.prototype.configure;

            nodeType.prototype.configure = function (info, ...args) {
                const result = configure.call(this, migrateLlmWidgetValues(this, info), ...args);
                refreshLlmProviderLabel(this);
                return result;
            };

            nodeType.prototype.onNodeCreated = function () {
                const r = onNodeCreated?.apply(this, arguments);

                const modelWidget = replaceModelTextWidgetWithCombo(this);

                this.aerinceModelRefreshButton = this.addWidget("button", "refresh_models", null, () => refreshLlmModels(this));
                if (nodeData.name === "LLMImagePromptReverseEditable") {
                    this.addWidget("button", "reverse_now", "反推", () => reverseImagePromptEditable(this));
                    const hashWidget = widgetByName(this, "source_image_hash");
                    if (hashWidget) {
                        // Keep serialized cache, hide from UI.
                        if (hashWidget.aerinceOrigComputeSize === undefined) {
                            hashWidget.aerinceOrigComputeSize = hashWidget.computeSize;
                        }
                        hashWidget.hidden = true;
                        hashWidget.computeSize = () => [0, -4];
                        if (typeof hashWidget.draw === "function" && hashWidget.aerinceOrigDraw === undefined) {
                            hashWidget.aerinceOrigDraw = hashWidget.draw;
                            hashWidget.draw = () => {};
                        }
                    }
                }

                const baseWidget = widgetByName(this, "base_url");
                const keyWidget = widgetByName(this, "api_key");
                const originalBaseCallback = baseWidget?.callback;
                const originalKeyCallback = keyWidget?.callback;
                const node = this;
                const resetModels = () => {
                    setModelOptions(modelWidget, []);
                    setThinkingModelDetails(node, []);
                    refreshThinkingWidgetLabel(node);
                };

                if (baseWidget) {
                    baseWidget.callback = function () {
                        originalBaseCallback?.apply(this, arguments);
                        resetModels();
                    };
                }
                if (keyWidget) {
                    keyWidget.callback = function () {
                        originalKeyCallback?.apply(this, arguments);
                        resetModels();
                    };
                }

                installLlmProviderUi(this, () => {
                    resetModels();
                    return refreshLlmModels(this, true);
                });
                refreshThinkingWidgetLabel(this);
                setTimeout(() => refreshLlmModels(this), 200);
                return r;
            };

            if (nodeData.name === "LLMImagePromptReverseEditable") {
                const onExecuted = nodeType.prototype.onExecuted;
                nodeType.prototype.onExecuted = function (message) {
                    onExecuted?.apply(this, arguments);
                    // When smart auto reverse runs, write returned prompt + image hash.
                    const values = message?.prompt_text;
                    if (Array.isArray(values) && typeof values[0] === "string" && values[0].trim()) {
                        setWidgetValue(widgetByName(this, "prompt_text"), values[0]);
                        const hashes = message?.source_image_hash;
                        if (Array.isArray(hashes) && typeof hashes[0] === "string" && hashes[0].trim()) {
                            setWidgetValue(widgetByName(this, "source_image_hash"), hashes[0]);
                        }
                    }
                    this.setDirtyCanvas?.(true, true);
                };
            }
        }
    },
};

app.registerExtension(ext);
