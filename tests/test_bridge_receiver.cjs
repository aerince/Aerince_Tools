// Offline regression: node tests/test_bridge_receiver.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

let extension;
let requests = 0;
let fetchHandler;
const alerts = [];
const app = {
    registerExtension(value) { extension = value; },
    graph: {
        links: { 1: { origin_id: 7 } },
        getNodeById() { return { imgs: ["data:image/png;base64,AA=="] }; },
        setDirtyCanvas() {},
    },
};
const source = fs.readFileSync(path.join(__dirname, "../js/bridge_receiver.js"), "utf8");
vm.runInNewContext(source.replace(/^import .*;\r?$/gm, ""), {
    app,
    installLlmProviderUi(node, onSwitch) { node.switchProvider = onSwitch; },
    refreshLlmProviderLabel(node) { node.providerLabelRestored = true; },
    api: { fetchApi(...args) { requests++; return fetchHandler(...args); } },
    alert(message) { alerts.push(message); },
    console: { log() {}, error() {} },
    setTimeout() {}, // Never run model refresh or background setup in this test.
    fetch() { throw new Error("Unexpected network request"); },
});

const schemas = {
    LLMImagePromptReverse: ["base_url", "api_key", "model", "system_prompt", "user_prompt", "temperature", "timeout", "thinking_strength"],
    LLMPromptOptimizer: ["prompt", "base_url", "api_key", "model", "system_prompt", "user_prompt", "temperature", "timeout", "thinking_strength"],
    LLMImagePromptReverseEditable: ["base_url", "api_key", "model", "system_prompt", "user_prompt", "prompt_text", "auto_reverse", "source_image_hash", "temperature", "timeout", "thinking_strength"],
};
const values = {
    base_url: "https://example.invalid/v1", api_key: "", model: "test-model",
    prompt: "input", system_prompt: "system", user_prompt: "user", prompt_text: "old prompt",
    auto_reverse: true, source_image_hash: "old-hash", temperature: 0.4,
    max_tokens: 250, timeout: 700, thinking_strength: "high",
};
const widget = (node, name) => node.widgets.find((item) => item.name === name);

async function createNode(type, combo = false) {
    class Node {
        constructor() {
            this.widgets = schemas[type].map((name) => ({
                name, value: values[name], type: name === "model" && combo ? "combo" : "text",
                options: {},
            }));
            this.inputs = [{ name: "image", link: 1 }];
            this.onNodeCreated();
        }
        addWidget(type, name, value, callback, options) {
            const item = { type, name, value, callback, options };
            this.widgets.push(item);
            return item;
        }
        configure(info) {
            this.configured = info;
            info.widgets_values?.forEach((value, index) => {
                if (this.widgets[index]) this.widgets[index].value = value;
            });
            return "configured";
        }
        setDirtyCanvas() {}
    }
    await extension.beforeRegisterNodeDef(Node, { name: type }, app);
    return new Node();
}

async function main() {
    let migrationCases = 0;
    for (const type of Object.keys(schemas)) {
        for (const combo of [false, true]) {
            for (const buttons of [false, true]) {
                for (const legacy of [false, true]) {
                    const node = await createNode(type, combo);
                    const names = [...schemas[type]];
                    if (!combo) names.splice(3, 0, ...names.splice(names.indexOf("model"), 1));
                    assert.deepEqual(node.widgets.filter((item) => item.type !== "button").map((item) => item.name), names);
                    const oldNames = [...names];
                    if (legacy) oldNames.splice(oldNames.indexOf("timeout"), 0, "max_tokens");
                    const tail = buttons ? node.widgets.filter((item) => item.type === "button").map((item) => item.value) : [];
                    const saved = { widgets_values: [...oldNames.map((name) => values[name]), ...tail] };
                    const original = JSON.stringify(saved);
                    assert.equal(node.configure(saved), "configured");
                    assert.equal(node.providerLabelRestored, true);
                    assert.equal(JSON.stringify(saved), original, "do not mutate the saved workflow");
                    assert.deepEqual(Array.from(node.configured.widgets_values), [...names.map((name) => values[name]), ...tail]);
                    for (const name of names) assert.equal(widget(node, name).value, values[name], `${type}: ${name}`);
                    const migrated = node.configured;
                    node.configure(migrated);
                    assert.equal(node.configured, migrated, "new layout and reloading must not remove another value");
                    migrationCases++;
                }
            }
        }
    }

    const type = "LLMImagePromptReverseEditable";
    const node = await createNode(type);
    const button = widget(node, "reverse_now");
    const other = await createNode(type);
    const pending = [];
    fetchHandler = (url, options) => {
        assert.equal(url, "/aerince/llm/reverse-image");
        const payload = JSON.parse(options.body);
        assert.equal("max_tokens" in payload, false);
        assert.equal(payload.timeout, 700);
        assert.equal(payload.thinking_strength, "high");
        return new Promise((resolve) => pending.push(resolve));
    };
    const first = button.callback();
    const duplicate = button.callback();
    const independent = widget(other, "reverse_now").callback();
    await Promise.resolve();
    assert.equal(requests, 2, "one request per node, no global lock");
    assert.equal(button.name, "反推中...");
    for (const resolve of pending) resolve({ ok: true, json: async () => ({ success: true, prompt: "new prompt", image_hash: "new-hash" }) });
    await Promise.all([first, duplicate, independent]);
    for (const current of [node, other]) {
        assert.equal(widget(current, "prompt_text").value, "new prompt");
        assert.equal(widget(current, "source_image_hash").value, "new-hash");
        assert.equal(current.aerinceReverseInFlight, false);
        assert.ok(widget(current, "reverse_now"));
    }

    const failures = [
        async () => { throw new Error("offline"); },
        async () => ({ ok: true, json: async () => { throw new SyntaxError("invalid JSON"); } }),
        async () => ({ ok: false, status: 500, json: async () => ({ success: true, prompt: "must not overwrite", image_hash: "wrong" }) }),
        ...[null, { success: false, error: "failed" }, ...[undefined, "", " \n\t", 42, {}].map((prompt) => ({ success: true, prompt, image_hash: "wrong" }))]
            .map((data) => async () => ({ ok: true, json: async () => data })),
    ];
    for (const fail of failures) {
        fetchHandler = fail;
        await button.callback();
        assert.equal(widget(node, "prompt_text").value, "new prompt");
        assert.equal(widget(node, "source_image_hash").value, "new-hash");
        assert.equal(node.aerinceReverseInFlight, false);
        assert.equal(button.name, "reverse_now");
    }
    node.inputs[0].link = null;
    const before = requests;
    await button.callback();
    assert.equal(requests, before, "image resolution failure must not send a request");
    assert.equal(node.aerinceReverseInFlight, false);
    assert.equal(button.name, "reverse_now");
    assert.equal(widget(node, "prompt_text").value, "new prompt");
    assert.equal(widget(node, "source_image_hash").value, "new-hash");
    assert.equal(alerts.length, failures.length + 1);
    node.inputs[0].link = 1;
    fetchHandler = async () => ({ ok: true, json: async () => ({ success: true, prompt: "retry", image_hash: "retry-hash" }) });
    await button.callback();
    assert.equal(widget(node, "prompt_text").value, "retry");
    assert.equal(widget(node, "source_image_hash").value, "retry-hash");

    for (const prompt of [undefined, "", " ", 42]) {
        node.onExecuted({ prompt_text: [prompt], source_image_hash: ["wrong"] });
        assert.equal(widget(node, "prompt_text").value, "retry");
        assert.equal(widget(node, "source_image_hash").value, "retry-hash");
    }
    node.onExecuted({ prompt_text: ["auto result"], source_image_hash: ["auto-hash"] });
    assert.equal(widget(node, "prompt_text").value, "auto result");
    assert.equal(widget(node, "source_image_hash").value, "auto-hash");
    // A stale refresh must never overwrite a provider selected while it was pending.
    const modelRequests = [];
    fetchHandler = (url, options) => {
        assert.equal(url, "/aerince/llm/models");
        modelRequests.push(JSON.parse(options.body));
        return new Promise((resolve) => pending.push(resolve));
    };
    pending.length = 0;
    const oldRefresh = node.aerinceModelRefreshButton.callback();
    widget(node, "base_url").value = "aerince-provider://new-provider";
    widget(node, "api_key").value = "";
    const newRefresh = node.switchProvider();
    assert.equal(widget(node, "model").value, "refresh_models_first");
    pending[1]({ ok: true, json: async () => ({ success: true, models: ["new-model"] }) });
    await newRefresh;
    pending[0]({ ok: true, json: async () => ({ success: true, models: ["stale-model"] }) });
    await oldRefresh;
    assert.equal(widget(node, "model").value, "new-model");
    assert.equal(node.aerinceModelRefreshButton.name, "refresh_models");
    assert.equal(modelRequests[1].base_url, "aerince-provider://new-provider");
    assert.equal(modelRequests[1].api_key, "");
    const previousAlerts = alerts.length;
    fetchHandler = async () => { throw new Error("offline models"); };
    await assert.rejects(node.switchProvider(), /offline models/);
    assert.equal(alerts.length, previousAlerts, "provider dialog handles refresh errors without alert");
    console.log(`PASS: ${migrationCases} widget layouts; duplicate/per-node guard; ${failures.length + 1} failures preserve state; retry/manual/auto success`);
    console.log("PASS: provider configure hook; switch refresh; stale response discarded; dialog error propagation");
}

main().catch((error) => { console.error(error); process.exitCode = 1; });
