import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const STYLE_ID = "aerince-lora-recipe-manager-style";
const MANAGED_PREFIX = "aerince_recipe_";
const UNCATED_TAG = "__uncategorized__";

function ensureStyle() {
    let style = document.getElementById(STYLE_ID);
    if (!style) {
        style = document.createElement("style");
        style.id = STYLE_ID;
        document.head.appendChild(style);
    }
    // Always refresh so UI polish can hot-reload after hard refresh.
    style.textContent = `
.aerince-modal-backdrop{position:fixed;inset:0;background:rgba(0,0,0,.58);z-index:10000;display:flex;align-items:center;justify-content:center;font-family:Arial,"Microsoft YaHei",sans-serif;backdrop-filter:blur(2px)}
.aerince-modal{--aer-radius:12px;--aer-radius-sm:8px;--aer-gap:12px;background:var(--comfy-menu-bg,#1e1f22);color:var(--fg-color,#e8e8e8);border:1px solid rgba(255,255,255,.08);border-radius:var(--aer-radius);box-shadow:0 18px 50px rgba(0,0,0,.5);max-width:960px;max-height:88vh;overflow:auto;padding:0;min-width:460px}
.aerince-modal-header{display:flex;align-items:center;justify-content:space-between;gap:12px;padding:16px 18px 12px;border-bottom:1px solid rgba(255,255,255,.06)}
.aerince-modal-header h2{margin:0;font-size:17px;font-weight:600;letter-spacing:.2px}
.aerince-modal-body{padding:16px 18px 18px}
.aerince-row{display:flex;gap:10px;align-items:center;margin:10px 0}
.aerince-row label{min-width:88px;color:var(--descrip-text,#a0a0a0);font-size:13px}
.aerince-modal input,.aerince-modal textarea,.aerince-modal select{background:var(--comfy-input-bg,#121316);color:var(--input-text,#eee);border:1px solid rgba(255,255,255,.1);border-radius:var(--aer-radius-sm);padding:8px 10px;outline:none}
.aerince-modal input:focus,.aerince-modal textarea:focus{border-color:rgba(59,130,246,.55);box-shadow:0 0 0 2px rgba(59,130,246,.15)}
.aerince-modal textarea{width:100%;min-height:80px}
.aerince-actions{display:flex;gap:8px;justify-content:flex-end;margin-top:16px}
.aerince-modal button{background:var(--comfy-accent,#3b82f6);color:#fff;border:0;border-radius:7px;padding:8px 12px;cursor:pointer;font-size:13px;transition:filter .12s ease,transform .08s ease}
.aerince-modal button:hover{filter:brightness(1.08)}
.aerince-modal button:active{transform:translateY(1px)}
.aerince-modal button:disabled{opacity:.55;cursor:default}
.aerince-modal button:disabled:hover{filter:none}
.aerince-modal button.secondary{background:rgba(255,255,255,.1);color:var(--fg-color,#ddd)}
.aerince-modal button.danger{background:#b91c1c}
.aerince-tabs{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px;padding-bottom:2px}
.aerince-chooser-toolbar{display:flex;gap:8px;margin-bottom:12px}
.aerince-chooser-toolbar input{flex:1;min-width:0}
.aerince-chooser-toolbar button{white-space:nowrap;background:rgba(255,255,255,.1);color:var(--fg-color,#ddd)}
.aerince-tab{background:rgba(255,255,255,.06);color:var(--descrip-text,#bbb);border:1px solid transparent;border-radius:999px;padding:6px 12px;cursor:pointer;font-size:12px;line-height:1.2}
.aerince-tab:hover{background:rgba(255,255,255,.1);color:var(--fg-color,#eee)}
.aerince-tab.active{background:rgba(59,130,246,.18);border-color:rgba(59,130,246,.45);color:#dbeafe}
.aerince-card-grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(158px,1fr));gap:14px;min-width:680px}
.aerince-card{background:rgba(255,255,255,.03);border:1px solid rgba(255,255,255,.08);border-radius:10px;overflow:hidden;display:flex;flex-direction:column;box-shadow:0 6px 16px rgba(0,0,0,.18);transition:border-color .12s ease,transform .12s ease}
.aerince-card:hover{border-color:rgba(255,255,255,.16);transform:translateY(-1px)}
.aerince-card-cover{height:150px;background:linear-gradient(180deg,#14151a,#0e0f12);display:flex;align-items:center;justify-content:center;color:#666;font-size:12px}
.aerince-card-cover img{width:100%;height:100%;object-fit:cover;display:block}
.aerince-card-name{padding:10px 10px 6px;text-align:center;font-size:13px;font-weight:500;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.aerince-card-buttons{display:flex;gap:6px;padding:0 8px 10px}
.aerince-card-buttons button{font-size:12px;padding:6px 6px;flex:1}
.aerince-small{font-size:12px;color:var(--descrip-text,#aaa)}
.aerince-empty{padding:28px 8px;text-align:center;color:var(--descrip-text,#888)}
.aerince-file-hint{margin-left:4px}
.aerince-chooser-list{max-height:58vh;overflow:auto;margin-top:10px;border:1px solid rgba(255,255,255,.06);border-radius:8px}
.aerince-chooser-item{padding:8px 10px;border-bottom:1px solid rgba(255,255,255,.06);cursor:pointer}
.aerince-chooser-item:hover{background:rgba(255,255,255,.07)}
.aerince-chooser-item:last-child{border-bottom:0}
`;
}

function widgetByName(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function hideWidget(widget) {
    if (!widget) return;
    // Community-standard hide: keep value/serialize, stop layout + drawing.
    if (widget.aerinceOrigType === undefined) widget.aerinceOrigType = widget.type;
    if (widget.aerinceOrigComputeSize === undefined) widget.aerinceOrigComputeSize = widget.computeSize;
    if (widget.aerinceOrigDraw === undefined && typeof widget.draw === "function") widget.aerinceOrigDraw = widget.draw;
    widget.hidden = true;
    widget.type = "converted-widget";
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};
}

function hideRawWidgets(node) {
    for (const name of ["lora_stack", "trigger_words", "recipe_id", "recipe_name"]) {
        hideWidget(widgetByName(node, name));
    }
}

function parseStack(node) {
    const widget = widgetByName(node, "lora_stack");
    try {
        const value = JSON.parse(widget?.value || "[]");
        return Array.isArray(value) ? value : [];
    } catch {
        return [];
    }
}

function normalizeStack(stack) {
    return (Array.isArray(stack) ? stack : [])
        .filter((item) => item && typeof item === "object")
        .map((item) => ({
            enabled: item.enabled !== false,
            name: String(item.name || item.lora || ""),
            strength: Number.isFinite(Number(item.strength)) ? Number(item.strength) : 1,
        }));
}

function writeStackWidget(node, stack) {
    const widget = widgetByName(node, "lora_stack");
    if (widget) {
        widget.value = JSON.stringify(normalizeStack(stack));
        widget.callback?.(widget.value);
    }
}

function syncStackFromWidgets(node) {
    const rowWidgets = (node.widgets || []).filter((widget) => widget.aerinceRowWidget);
    const stack = rowWidgets.map((widget) => normalizeStack([widget.value])[0]).filter(Boolean);
    writeStackWidget(node, stack);
    return stack;
}

function setWidgetValue(node, name, value) {
    const widget = widgetByName(node, name);
    if (widget) {
        widget.value = value ?? "";
        widget.callback?.(widget.value);
    }
}

function escapeHtml(value) {
    return String(value ?? "")
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

async function fetchJson(path, options) {
    const response = await api.fetchApi(path, options);
    const data = await response.json();
    if (!response.ok || !data.success) throw new Error(data.error || `HTTP ${response.status}`);
    return data;
}

async function getLoras(node) {
    if (!node.aerinceLoraOptions) {
        const data = await fetchJson("/aerince/lora-recipes/loras");
        node.aerinceLoraOptions = data.lora_details || data.loras || [];
    }
    return node.aerinceLoraOptions;
}

async function getRecipes() {
    const data = await fetchJson("/aerince/lora-recipes");
    return data.recipes || [];
}

function stackTriggerKey(stack) {
    // Strength does not affect trigger words; only membership/enabled/name does.
    return JSON.stringify(
        normalizeStack(stack).map((item) => [item.enabled !== false, item.name || ""]),
    );
}

async function extractTriggers(stack) {
    const data = await fetchJson("/aerince/lora-recipes/extract-triggers", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            loras: normalizeStack(stack),
            max_words_per_lora: 1,
            separator: ", ",
            use_civitai_fallback: true,
            filter_common_tags: true,
        }),
    });
    return data.trigger_words || "";
}

async function autoRefreshTriggers(node, stack = null, { force = false } = {}) {
    const current = normalizeStack(stack ?? syncStackFromWidgets(node));
    const key = stackTriggerKey(current);
    if (!force && node.aerinceTriggerKey === key) return;
    node.aerinceTriggerKey = key;
    const seq = (node.aerinceTriggerSeq = (node.aerinceTriggerSeq || 0) + 1);
    try {
        const triggers = await extractTriggers(current);
        if (seq !== node.aerinceTriggerSeq) return; // newer refresh won
        setWidgetValue(node, "trigger_words", triggers);
        node.setDirtyCanvas?.(true, true);
        app.canvas.setDirty(true, true);
    } catch (e) {
        console.warn("[AerinceTools] auto trigger refresh failed", e);
    }
}

function closeModal(backdrop) {
    backdrop?.remove();
}

function makeModal(title) {
    ensureStyle();
    const backdrop = document.createElement("div");
    backdrop.className = "aerince-modal-backdrop";
    const modal = document.createElement("div");
    modal.className = "aerince-modal";
    modal.innerHTML = `
<div class="aerince-modal-header"><h2>${escapeHtml(title)}</h2></div>
<div class="aerince-modal-body"></div>`;
    const body = modal.querySelector(".aerince-modal-body");
    backdrop.appendChild(modal);
    backdrop.addEventListener("click", (e) => { if (e.target === backdrop) closeModal(backdrop); });
    document.body.appendChild(backdrop);
    return { backdrop, modal, body };
}

function removeManagedWidgets(node) {
    if (!Array.isArray(node.widgets)) return;
    for (let i = node.widgets.length - 1; i >= 0; i--) {
        const widget = node.widgets[i];
        if (widget?.aerinceManaged) node.widgets.splice(i, 1);
    }
}

function markManaged(widget, index = null) {
    widget.aerinceManaged = true;
    widget.aerinceIndex = index;
    widget.options = widget.options || {};
    widget.options.serialize = false;
    return widget;
}

function addManagedButton(node, label, callback) {
    return markManaged(node.addWidget("button", label, null, callback));
}

function fitTextStart(ctx, text, maxWidth) {
    text = String(text || "");
    if (ctx.measureText(text).width <= maxWidth) return text;
    const ellipsis = "…";
    let out = text;
    while (out.length > 4 && ctx.measureText(out + ellipsis).width > maxWidth) out = out.slice(0, -1);
    return out + ellipsis;
}

const ROOT_DIR_KEY = "";

function loraNameOf(item) {
    return typeof item === "string" ? item : String(item?.name || "");
}

function loraCoverOf(item) {
    return item && typeof item === "object" ? String(item.cover || "") : "";
}

function loraCreatedAt(item) {
    return item && typeof item === "object" ? Number(item.created_at || 0) : 0;
}

function loraNames(list) {
    return (Array.isArray(list) ? list : []).map(loraNameOf).filter(Boolean);
}

function loraDir(name) {
    const parts = String(name || "").replace(/\\/g, "/").split("/");
    return parts.length > 1 ? parts[0] : ROOT_DIR_KEY;
}

function loraBasename(name) {
    return String(name || "").replace(/\\/g, "/").split("/").pop();
}

function loraKey(name) {
    return String(name || "").replace(/\\/g, "/").toLowerCase();
}

function coverUrl(url) {
    url = String(url || "");
    return /^(?:https?:)?\/\//i.test(url) || /^(?:data|blob):/i.test(url) ? url : api.apiURL(url);
}

const CHOOSER_SORT_KEY = "aerince_lora_chooser_sort";

function loadChooserSortMode(max) {
    try {
        const saved = Number(localStorage.getItem(CHOOSER_SORT_KEY));
        return Number.isInteger(saved) && saved >= 0 && saved < max ? saved : 0;
    } catch {
        return 0;
    }
}

function saveChooserSortMode(mode) {
    try {
        localStorage.setItem(CHOOSER_SORT_KEY, String(mode));
    } catch {
        // localStorage unavailable (privacy mode) — sort stays per-session only.
    }
}

function openLoraChooserModal(rowWidget) {
    const { backdrop, body } = makeModal("选择 LoRA");
    let loras = [];
    let tabs = [];
    let activeTab = "all";

    // Recomputed on every (re)fresh so tab counts follow the live LoRA list.
    function recomputeTabs() {
        loras = (rowWidget.loraOptions || []).filter((l) => loraNameOf(l));
        const dirCounts = new Map();
        for (const l of loras) {
            const key = loraDir(loraNameOf(l));
            dirCounts.set(key, (dirCounts.get(key) || 0) + 1);
        }
        const namedDirs = [...dirCounts.keys()].filter((d) => d !== ROOT_DIR_KEY).sort((a, b) => a.localeCompare(b, "zh"));
        tabs = [{ key: "all", dir: null, label: "全部", count: loras.length }];
        for (const d of namedDirs) tabs.push({ key: `dir:${d}`, dir: d, label: d, count: dirCounts.get(d) });
        if (dirCounts.has(ROOT_DIR_KEY)) tabs.push({ key: "dir:", dir: ROOT_DIR_KEY, label: "根目录", count: dirCounts.get(ROOT_DIR_KEY) });
        if (!tabs.some((tab) => tab.key === activeTab)) activeTab = "all";
    }

    const search = document.createElement("input");
    search.placeholder = "搜索 LoRA 文件名...";
    search.style.boxSizing = "border-box";
    const sortButton = document.createElement("button");
    sortButton.type = "button";
    const refreshButton = document.createElement("button");
    refreshButton.type = "button";
    refreshButton.textContent = "刷新列表";
    refreshButton.onclick = async () => {
        refreshButton.disabled = true;
        try {
            const data = await fetchJson(`/aerince/lora-recipes/loras?_=${Date.now()}`);
            const fresh = data.lora_details || data.loras || [];
            const node = rowWidget.aerinceNode;
            if (node) {
                node.aerinceLoraOptions = fresh;
                const known = new Set(loraNames(fresh).map(loraKey));
                for (const w of node.widgets || []) {
                    if (!w?.aerinceRowWidget) continue;
                    const current = w.value?.name || "";
                    w.loraOptions = current && !known.has(loraKey(current)) ? [{ name: current, cover: "" }, ...fresh] : fresh;
                }
            } else {
                rowWidget.loraOptions = fresh;
            }
            recomputeTabs();
            renderTabs();
            renderCards();
        } catch (e) {
            alert(`刷新失败：${e.message || e}`);
        } finally {
            refreshButton.disabled = false;
        }
    };
    const toolbar = document.createElement("div");
    toolbar.className = "aerince-chooser-toolbar";
    toolbar.append(search, sortButton, refreshButton);
    const tabBar = document.createElement("div");
    tabBar.className = "aerince-tabs";
    const grid = document.createElement("div");
    grid.className = "aerince-card-grid";
    grid.style.minWidth = "0";
    body.append(toolbar, tabBar, grid);

    let query = "";
    const sortLabels = ["排序：默认", "排序：名称 ↑", "排序：名称 ↓", "排序：创建时间（最新）", "排序：创建时间（最早）"];
    let sortMode = loadChooserSortMode(sortLabels.length);
    const collator = new Intl.Collator("zh", { numeric: true, sensitivity: "base" });

    function filtered() {
        const activeDir = tabs.find((tab) => tab.key === activeTab)?.dir;
        const items = loras.filter((l) => {
            const name = loraNameOf(l);
            if (activeDir !== null && loraDir(name) !== activeDir) return false;
            if (query && !name.toLowerCase().includes(query)) return false;
            return true;
        });
        if (sortMode) items.sort((a, b) => {
            const byName = collator.compare(loraBasename(loraNameOf(a)), loraBasename(loraNameOf(b)))
                || collator.compare(loraNameOf(a), loraNameOf(b));
            if (sortMode === 1) return byName;
            if (sortMode === 2) return -byName;
            const byCreated = loraCreatedAt(a) - loraCreatedAt(b);
            return sortMode === 3 ? -byCreated || byName : byCreated || byName;
        });
        return items;
    }

    function renderTabs() {
        tabBar.innerHTML = "";
        for (const tab of tabs) {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = `aerince-tab${tab.key === activeTab ? " active" : ""}`;
            btn.textContent = `${tab.label} (${tab.count})`;
            btn.onclick = () => {
                activeTab = tab.key;
                renderTabs();
                renderCards();
            };
            tabBar.appendChild(btn);
        }
    }

    function renderCards() {
        const items = filtered();
        grid.innerHTML = "";
        if (!items.length) {
            grid.innerHTML = `<div class="aerince-empty">没有匹配的 LoRA</div>`;
            return;
        }
        for (const l of items) {
            const name = loraNameOf(l);
            const cover = loraCoverOf(l);
            const card = document.createElement("div");
            card.className = "aerince-card";
            card.style.cursor = "pointer";
            card.innerHTML = `
<div class="aerince-card-cover">${cover ? `<img src="${escapeHtml(coverUrl(cover))}" alt="" loading="lazy">` : "无封面"}</div>
<div class="aerince-card-name" title="${escapeHtml(name)}">${escapeHtml(loraBasename(name))}</div>`;
            card.querySelector("img")?.addEventListener("error", (event) => {
                event.currentTarget.parentElement.textContent = "无封面";
            });
            card.onclick = () => {
                rowWidget.value.name = name;
                rowWidget.onChange?.();
                closeModal(backdrop);
            };
            grid.appendChild(card);
        }
    }

    search.oninput = () => {
        query = search.value.trim().toLowerCase();
        renderCards();
    };
    sortButton.onclick = () => {
        sortMode = (sortMode + 1) % sortLabels.length;
        saveChooserSortMode(sortMode);
        sortButton.textContent = sortLabels[sortMode];
        renderCards();
    };
    sortButton.textContent = sortLabels[sortMode];
    recomputeTabs();
    renderTabs();
    renderCards();
    setTimeout(() => search.focus(), 0);
}

class AerinceRecipeNameWidget {
    constructor(getName) {
        this.name = `${MANAGED_PREFIX}recipe_label`;
        this.type = "custom";
        this.options = { serialize: false };
        this.aerinceManaged = true;
        this.aerinceRecipeLabel = true;
        this.getName = getName;
        this.value = "";
    }

    computeSize() {
        return [0, LiteGraph.NODE_WIDGET_HEIGHT + 6];
    }

    draw(ctx, node, width, y) {
        const h = LiteGraph.NODE_WIDGET_HEIGHT + 2;
        const margin = 10;
        const nodeW = Number(node?.size?.[0]);
        const raw = Number.isFinite(nodeW) && nodeW > 0 ? nodeW : (Number(width) > 0 ? Number(width) : 320);
        const rowW = Math.max(120, raw - margin * 2);
        const name = String(this.getName?.() || "").trim();
        const label = name || "未选择画风";
        ctx.save();
        ctx.globalAlpha = app.canvas.editor_alpha;
        // Soft panel, separate from LoRA rows — never overlaps them.
        ctx.fillStyle = "rgba(59,130,246,0.10)";
        ctx.strokeStyle = "rgba(59,130,246,0.28)";
        const path = new Path2D();
        if (path.roundRect) path.roundRect(margin, y + 1, rowW, h - 2, 8);
        else path.rect(margin, y + 1, rowW, h - 2);
        ctx.fill(path);
        ctx.stroke(path);
        ctx.fillStyle = name ? LiteGraph.WIDGET_TEXT_COLOR : LiteGraph.WIDGET_SECONDARY_TEXT_COLOR;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = name ? "bold 12px Arial" : "12px Arial";
        // Name only, centered — no "画风" prefix.
        ctx.fillText(fitTextStart(ctx, label, rowW - 20), margin + rowW / 2, y + h / 2);
        ctx.restore();
    }

    mouse() {
        return false;
    }
}

class AerinceLoraRowWidget {
    constructor(name, value, loraOptions, onChange, onDelete, node) {
        this.name = name;
        this.type = "custom";
        this.options = { serialize: false };
        this.aerinceManaged = true;
        this.aerinceRowWidget = true;
        this.aerinceNode = node || null;
        this.value = normalizeStack([value])[0] || { enabled: true, name: "", strength: 1 };
        this.loraOptions = loraOptions || [];
        this.onChange = onChange;
        this.onDelete = onDelete;
        this.hitAreas = {};
        this.draggingStrength = false;
        this.strengthPointerActive = false;
        this.strengthMoved = false;
        this.dragStartX = 0;
        this.dragStartStrength = 1;
        this.lastDrawWidth = 0;
    }

    _usableWidth(node, width) {
        const nodeW = Number(node?.size?.[0]);
        const raw = Number.isFinite(nodeW) && nodeW > 0
            ? nodeW
            : (Number(width) > 0 ? Number(width) : (this.lastDrawWidth || 320));
        const margin = 10;
        return Math.max(120, raw - margin * 2);
    }

    computeSize() {
        return [0, LiteGraph.NODE_WIDGET_HEIGHT + 4];
    }

    draw(ctx, node, width, y) {
        const h = LiteGraph.NODE_WIDGET_HEIGHT;
        const margin = 10;
        const gap = 5;
        const midY = y + h / 2;
        const x0 = margin;
        const rowW = this._usableWidth(node, width);
        this.lastDrawWidth = rowW + margin * 2;
        ctx.save();
        ctx.globalAlpha = app.canvas.editor_alpha;
        ctx.fillStyle = LiteGraph.WIDGET_BGCOLOR;
        ctx.strokeStyle = "rgba(255,255,255,0.10)";
        const path = new Path2D();
        const x = x0, yy = y + 2, w = rowW, hh = h - 4;
        if (path.roundRect) path.roundRect(x, yy, w, hh, 8);
        else path.rect(x, yy, w, hh);
        ctx.fill(path);
        ctx.stroke(path);

        let posX = x0 + 6;
        const delW = Math.max(14, Math.min(18, rowW * 0.055));
        this.hitAreas.delete = [posX, y, delW, h];
        ctx.fillStyle = "#d16b6b";
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.fillText("×", posX + delW / 2, midY);
        posX += delW + gap;

        const toggleW = Math.max(14, Math.min(18, rowW * 0.055));
        this.hitAreas.toggle = [posX, y, toggleW, h];
        const boxX = posX + Math.max(1, (toggleW - 11) / 2);
        const boxY = y + (h - 11) / 2;
        ctx.strokeStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.strokeRect(boxX, boxY, 11, 11);
        if (this.value.enabled !== false) {
            ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
            ctx.fillText("✓", posX + toggleW / 2, midY + 0.5);
        }
        posX += toggleW + gap;

        const strengthW = Math.max(62, Math.min(82, rowW * 0.24));
        const loraW = Math.max(40, x0 + rowW - posX - strengthW - gap - 6);
        this.hitAreas.lora = [posX, y, loraW, h];
        ctx.fillStyle = this.value.enabled === false ? LiteGraph.WIDGET_SECONDARY_TEXT_COLOR : LiteGraph.WIDGET_TEXT_COLOR;
        ctx.textAlign = "left";
        const label = this.value.name || "选择 LoRA";
        ctx.fillText(fitTextStart(ctx, label, loraW - 2), posX, midY);
        posX += loraW + gap;

        const sideW = Math.max(13, Math.min(16, strengthW * 0.22));
        const valW = Math.max(30, strengthW - sideW * 2);
        this.hitAreas.strengthDec = [posX, y, sideW, h];
        this.hitAreas.strengthVal = [posX + sideW, y, valW, h];
        this.hitAreas.strengthInc = [posX + sideW + valW, y, sideW, h];
        this.hitAreas.strengthDrag = [posX, y, strengthW, h];
        const valX = posX + sideW;
        ctx.strokeStyle = "rgba(255,255,255,0.18)";
        ctx.beginPath();
        ctx.moveTo(valX + 3, y + h - 6);
        ctx.lineTo(valX + valW - 3, y + h - 6);
        ctx.stroke();
        ctx.textAlign = "center";
        ctx.fillStyle = LiteGraph.WIDGET_TEXT_COLOR;
        ctx.fillText("−", posX + sideW / 2, midY);
        ctx.fillText(Number(this.value.strength ?? 1).toFixed(2).replace(/\.00$/, ""), posX + sideW + valW / 2, midY);
        ctx.fillText("+", posX + sideW + valW + sideW / 2, midY);
        ctx.restore();
    }

    _hit(pos, area) {
        return area && pos[0] >= area[0] && pos[0] <= area[0] + area[2] && pos[1] >= area[1] && pos[1] <= area[1] + area[3];
    }

    _promptStrength(event) {
        const current = this.value.strength ?? 1;
        app.canvas.prompt("强度 strength", current, (v) => {
            const n = Number(v);
            if (!Number.isFinite(n)) return;
            this.value.strength = Math.round(n * 100) / 100;
            this.onChange?.();
        }, event);
    }

    mouse(event, pos) {
        if (event.type === "pointermove" && this.strengthPointerActive) {
            const dx = Number(event.deltaX ?? (pos[0] - this.dragStartX) ?? 0);
            if (Number.isFinite(dx) && Math.abs(dx) >= 2) {
                if (!this.draggingStrength) {
                    this.draggingStrength = true;
                    this.strengthMoved = true;
                }
            }
            if (this.draggingStrength && Number.isFinite(dx) && dx !== 0) {
                this.value.strength = Math.round((this.dragStartStrength + (pos[0] - this.dragStartX) * 0.01) * 100) / 100;
                this.onChange?.();
            }
            return true;
        }
        if (event.type === "pointerup") {
            if (this.strengthPointerActive) {
                const wasDrag = this.draggingStrength || this.strengthMoved;
                this.strengthPointerActive = false;
                this.draggingStrength = false;
                this.strengthMoved = false;
                if (!wasDrag && this._hit(pos, this.hitAreas.strengthVal)) {
                    this._promptStrength(event);
                    return true;
                }
            }
            this.draggingStrength = false;
            this.strengthPointerActive = false;
            return false;
        }
        if (event.type !== "pointerdown") return false;
        if (this._hit(pos, this.hitAreas.delete)) {
            this.onDelete?.();
            return true;
        }
        if (this._hit(pos, this.hitAreas.toggle)) {
            this.value.enabled = this.value.enabled === false;
            this.onChange?.();
            return true;
        }
        if (this._hit(pos, this.hitAreas.strengthDec)) {
            this.value.strength = Math.round((Number(this.value.strength ?? 1) - 0.05) * 100) / 100;
            this.onChange?.();
            return true;
        }
        if (this._hit(pos, this.hitAreas.strengthInc)) {
            this.value.strength = Math.round((Number(this.value.strength ?? 1) + 0.05) * 100) / 100;
            this.onChange?.();
            return true;
        }
        if (this._hit(pos, this.hitAreas.strengthVal)) {
            this.strengthPointerActive = true;
            this.strengthMoved = false;
            this.draggingStrength = false;
            this.dragStartX = pos[0];
            this.dragStartStrength = Number(this.value.strength ?? 1);
            if (event.detail >= 2) {
                this.strengthPointerActive = false;
                this._promptStrength(event);
            }
            return true;
        }
        if (this._hit(pos, this.hitAreas.lora)) {
            openLoraChooserModal(this);
            return true;
        }
        return false;
    }
}

function currentRecipeName(node) {
    return (
        widgetByName(node, "recipe_name")?.value
        || node.aerinceCurrentRecipe?.name
        || ""
    );
}

async function rebuildNodeWidgets(node, stack = null) {
    hideRawWidgets(node);
    const oldSize = node.size ? [...node.size] : [300, 200];
    const loras = await getLoras(node).catch((e) => {
        console.error("[AerinceTools] failed to load LoRA list", e);
        return [];
    });
    stack = normalizeStack(stack ?? parseStack(node));
    writeStackWidget(node, stack);
    removeManagedWidgets(node);
    const knownLoras = new Set(loraNames(loras).map(loraKey));

    // Dedicated non-overlapping recipe name row (display only).
    const nameWidget = new AerinceRecipeNameWidget(() => currentRecipeName(node));
    if (node.addCustomWidget) node.addCustomWidget(nameWidget);
    else node.widgets.push(nameWidget);

    stack.forEach((item, index) => {
        const row = new AerinceLoraRowWidget(
            `${MANAGED_PREFIX}row_${index}`,
            item,
            item.name && !knownLoras.has(loraKey(item.name)) ? [{ name: item.name, cover: "" }, ...loras] : loras,
            () => {
                const next = syncStackFromWidgets(node);
                node.setDirtyCanvas?.(true, true);
                app.canvas.setDirty(true, true);
                // Auto-refresh when LoRA set changes (add/remove/rename/toggle), not strength drag.
                autoRefreshTriggers(node, next);
            },
            () => {
                const next = syncStackFromWidgets(node);
                next.splice(index, 1);
                rebuildNodeWidgets(node, next);
            },
            node,
        );
        if (node.addCustomWidget) node.addCustomWidget(row);
        else node.widgets.push(row);
    });

    addManagedButton(node, "+ 添加 LoRA", async () => {
        const current = syncStackFromWidgets(node);
        const options = await getLoras(node).catch(() => []);
        current.push({ enabled: true, name: loraNameOf(options[0]), strength: 1 });
        rebuildNodeWidgets(node, current);
    });

    addManagedButton(node, "新建画风", () => {
        if (!confirm("新建画风会清空当前节点的画风绑定和 LoRA 列表（不影响已保存的配方），确定继续？")) return;
        startNewRecipe(node).catch((e) => alert(`新建画风失败：${e.message || e}`));
    });
    addManagedButton(node, "保存画风", () => openSaveDialog(node));
    addManagedButton(node, "打开管理器", () => openManager(node).catch((e) => alert(`打开管理器失败：${e.message || e}`)));

    const computed = node.computeSize?.() || oldSize;
    node.size = [
        Math.max(oldSize[0] || 320, computed[0] || 320, 320),
        Math.max(200, computed[1] || oldSize[1] || 200),
    ];
    node.setDirtyCanvas?.(true, true);
    app.canvas.setDirty(true, true);
    // Keep trigger_words in sync after rebuild (add/remove/load recipe/stack restore).
    autoRefreshTriggers(node, stack, { force: true });
}

async function uploadCover(file) {
    if (!file) return { cover: "", cover_url: "" };
    const form = new FormData();
    form.append("file", file);
    return await fetchJson("/aerince/lora-recipes/cover", { method: "POST", body: form });
}

async function resolveCurrentRecipe(node) {
    // recipe_id survives workflow reloads but the in-memory recipe copy does not;
    // re-fetch from disk so overwrite saves inherit cover/tags instead of wiping them.
    const currentId = widgetByName(node, "recipe_id")?.value || "";
    let currentRecipe = node.aerinceCurrentRecipe;
    if (currentId && (!currentRecipe || currentRecipe.id !== currentId)) {
        try {
            currentRecipe = (await getRecipes()).find((r) => r.id === currentId) || {};
        } catch (e) {
            console.warn("[AerinceTools] failed to resolve current recipe", e);
            currentRecipe = currentRecipe || {};
        }
        node.aerinceCurrentRecipe = currentRecipe;
    }
    return currentRecipe || {};
}

async function openSaveDialog(node) {
    const stack = syncStackFromWidgets(node);
    let autoTrigger = widgetByName(node, "trigger_words")?.value || "";
    if (!autoTrigger) {
        try { autoTrigger = await extractTriggers(stack); } catch (e) { console.warn(e); }
    }
    const { backdrop, body } = makeModal("保存画风");
    const currentId = widgetByName(node, "recipe_id")?.value || "";
    const currentRecipe = await resolveCurrentRecipe(node);
    const nameVal = escapeHtml(widgetByName(node, "recipe_name")?.value || currentRecipe.name || "");
    const descVal = escapeHtml(currentRecipe.description || "");
    const tagsVal = escapeHtml((currentRecipe.model_tags || []).join(", "));
    body.insertAdjacentHTML("beforeend", `
<div class="aerince-row"><label>画风名</label><input id="aerince-name" style="flex:1" value="${nameVal}"></div>
<div class="aerince-row"><label>描述</label><input id="aerince-desc" style="flex:1" value="${descVal}"></div>
<div class="aerince-row"><label>模型标签</label><input id="aerince-tags" style="flex:1" placeholder="anima, SDXL, Illustrious" value="${tagsVal}"></div>
<div class="aerince-row"><label>封面</label><input id="aerince-cover" type="file" accept="image/*"><span class="aerince-small aerince-file-hint">不选则保留原封面</span></div>
<div class="aerince-actions">
  <button class="secondary" id="aerince-cancel">取消</button>
  ${currentId ? '<button class="secondary" id="aerince-save-as">另存为</button>' : ""}
  <button id="aerince-save">${currentId ? "覆盖保存" : "保存"}</button>
</div>`);
    body.querySelector("#aerince-cancel").onclick = () => closeModal(backdrop);

    async function doSave(asNew) {
        const name = body.querySelector("#aerince-name").value.trim();
        if (!name) return alert("请填写画风名");
        const coverInfo = await uploadCover(body.querySelector("#aerince-cover").files?.[0]);
        const id = asNew || !currentId ? "" : currentId;
        const recipe = {
            id,
            name,
            description: body.querySelector("#aerince-desc").value.trim(),
            cover: coverInfo.cover || currentRecipe.cover || "",
            model_tags: body.querySelector("#aerince-tags").value.split(",").map((x) => x.trim()).filter(Boolean),
            trigger_words: autoTrigger,
            loras: stack,
        };
        const saved = await fetchJson("/aerince/lora-recipes/save", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ recipe, overwrite: true }),
        });
        setWidgetValue(node, "recipe_id", saved.recipe.id);
        setWidgetValue(node, "recipe_name", saved.recipe.name);
        setWidgetValue(node, "trigger_words", saved.recipe.trigger_words || "");
        node.aerinceCurrentRecipe = saved.recipe;
        await rebuildNodeWidgets(node, saved.recipe.loras || stack);
        closeModal(backdrop);
    }
    body.querySelector("#aerince-save").onclick = () => doSave(false).catch((e) => alert(`保存失败：${e.message || e}`));
    const saveAs = body.querySelector("#aerince-save-as");
    if (saveAs) saveAs.onclick = () => doSave(true).catch((e) => alert(`保存失败：${e.message || e}`));
}

function recipeTags(recipe) {
    const tags = Array.isArray(recipe?.model_tags)
        ? recipe.model_tags.map((t) => String(t || "").trim()).filter(Boolean)
        : [];
    return tags.length ? tags : [UNCATED_TAG];
}

function collectModelTabs(recipes) {
    const counts = new Map();
    for (const recipe of recipes) {
        for (const tag of recipeTags(recipe)) {
            counts.set(tag, (counts.get(tag) || 0) + 1);
        }
    }
    const named = [...counts.keys()]
        .filter((t) => t !== UNCATED_TAG)
        .sort((a, b) => a.localeCompare(b, "zh"));
    const tabs = [{ id: "all", label: "全部", count: recipes.length }];
    for (const tag of named) tabs.push({ id: tag, label: tag, count: counts.get(tag) || 0 });
    if (counts.has(UNCATED_TAG)) {
        tabs.push({ id: UNCATED_TAG, label: "未分类", count: counts.get(UNCATED_TAG) || 0 });
    }
    return tabs;
}

function filterRecipesByTab(recipes, tabId) {
    if (!tabId || tabId === "all") return recipes;
    if (tabId === UNCATED_TAG) {
        return recipes.filter((r) => !Array.isArray(r.model_tags) || !r.model_tags.some((t) => String(t || "").trim()));
    }
    return recipes.filter((r) => recipeTags(r).includes(tabId));
}

async function startNewRecipe(node) {
    // Unbind the node from its recipe and start from a blank palette; the next
    // 保存画风 then creates a brand-new recipe instead of overwriting the old one.
    setWidgetValue(node, "recipe_id", "");
    setWidgetValue(node, "recipe_name", "");
    setWidgetValue(node, "trigger_words", "");
    node.aerinceCurrentRecipe = null;
    await rebuildNodeWidgets(node, []);
}

async function openManager(node, preferredTab = null) {
    const { backdrop, body } = makeModal("LoRA 画风管理器");
    const recipes = await getRecipes();
    const tabs = collectModelTabs(recipes);
    let activeTab = preferredTab && tabs.some((t) => t.id === preferredTab)
        ? preferredTab
        : (node.aerinceManagerTab && tabs.some((t) => t.id === node.aerinceManagerTab) ? node.aerinceManagerTab : "all");

    const tabBar = document.createElement("div");
    tabBar.className = "aerince-tabs";
    const grid = document.createElement("div");
    grid.className = "aerince-card-grid";
    const actions = document.createElement("div");
    actions.className = "aerince-actions";
    const close = document.createElement("button");
    close.className = "secondary";
    close.textContent = "关闭";
    close.onclick = () => closeModal(backdrop);
    actions.appendChild(close);
    body.append(tabBar, grid, actions);

    function applyRecipe(recipe, editAfter = false) {
        return async () => {
            setWidgetValue(node, "recipe_id", recipe.id || "");
            setWidgetValue(node, "recipe_name", recipe.name || "");
            setWidgetValue(node, "trigger_words", recipe.trigger_words || "");
            node.aerinceCurrentRecipe = recipe;
            await rebuildNodeWidgets(node, recipe.loras || []);
            closeModal(backdrop);
            if (editAfter) openSaveDialog(node);
        };
    }

    async function deleteRecipe(recipe) {
        if (!confirm(`删除画风配方「${recipe.name}」？`)) return;
        await fetchJson("/aerince/lora-recipes/delete", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ id: recipe.id }),
        });
        closeModal(backdrop);
        openManager(node, activeTab);
    }

    function renderTabs() {
        tabBar.innerHTML = "";
        for (const tab of tabs) {
            const btn = document.createElement("button");
            btn.type = "button";
            btn.className = `aerince-tab${tab.id === activeTab ? " active" : ""}`;
            btn.textContent = `${tab.label} (${tab.count})`;
            btn.onclick = () => {
                activeTab = tab.id;
                node.aerinceManagerTab = activeTab;
                renderTabs();
                renderCards();
            };
            tabBar.appendChild(btn);
        }
    }

    function renderCards() {
        const filtered = filterRecipesByTab(recipes, activeTab);
        grid.innerHTML = "";
        if (!filtered.length) {
            grid.innerHTML = `<div class="aerince-empty">${recipes.length ? "当前分类下暂无画风。" : "暂无保存的画风配方。"}</div>`;
            return;
        }
        for (const recipe of filtered) {
            const card = document.createElement("div");
            card.className = "aerince-card";
            const cover = recipe.cover_url
                ? `<img src="${escapeHtml(recipe.cover_url)}" alt="">`
                : "无封面";
            card.innerHTML = `
<div class="aerince-card-cover">${cover}</div>
<div class="aerince-card-name" title="${escapeHtml(recipe.name)}">${escapeHtml(recipe.name)}</div>
<div class="aerince-card-buttons">
  <button data-act="use">使用</button>
  <button class="secondary" data-act="edit">编辑</button>
  <button class="danger" data-act="delete">删除</button>
</div>`;
            card.querySelector('[data-act="use"]').onclick = applyRecipe(recipe, false);
            card.querySelector('[data-act="edit"]').onclick = applyRecipe(recipe, true);
            card.querySelector('[data-act="delete"]').onclick = () => deleteRecipe(recipe).catch((e) => alert(`删除失败：${e.message || e}`));
            grid.appendChild(card);
        }
    }

    renderTabs();
    renderCards();
}

app.registerExtension({
    name: "Aerince.LoRARecipeManager",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AerinceLoRARecipeManager") return;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const r = onNodeCreated?.apply(this, arguments);
            ensureStyle();
            hideRawWidgets(this);
            setTimeout(() => rebuildNodeWidgets(this).catch((e) => console.error("[AerinceTools] LoRA recipe UI failed", e)), 0);
            return r;
        };
        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const r = onConfigure?.apply(this, arguments);
            setTimeout(() => rebuildNodeWidgets(this).catch((e) => console.error("[AerinceTools] LoRA recipe UI configure failed", e)), 0);
            return r;
        };
    },
});
