// Offline behavior check: node tests/test_llm_provider_manager.cjs
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

class Element {
    constructor(tag) {
        this.tagName = tag;
        this.children = [];
        this.style = {};
        this.dataset = {};
        this.attributes = {};
        this.listeners = {};
        this.value = "";
        this.checked = false;
    }
    append(...items) { for (const item of items) { item.parent = this; this.children.push(item); } }
    replaceChildren(...items) { this.children = []; this.append(...items); }
    setAttribute(name, value) { this.attributes[name] = value; }
    addEventListener(name, callback) { this.listeners[name] = callback; }
    focus() { document.activeElement = this; }
    showModal() { this.open = true; }
    close() { this.open = false; this.listeners.close?.(); }
    remove() { this.parent.children = this.parent.children.filter((item) => item !== this); }
    reportValidity() { return true; }
    set innerHTML(value) { throw new Error("Do not inject HTML"); }
}
const document = { createElement: (tag) => new Element(tag), body: new Element("body") };
let fetchHandler;
const requests = [];
const context = vm.createContext({ document, api: { fetchApi(url, options) {
    requests.push({ url, options });
    return fetchHandler(url, options);
} } });
const source = fs.readFileSync(path.join(__dirname, "../js/llm_provider_manager.js"), "utf8");
vm.runInContext(source.replace(/^import .*;\r?$/gm, "").replace(/^export /gm, "") +
    "\nglobalThis.ui = {installLlmProviderUi, refreshLlmProviderLabel};", context);
const { installLlmProviderUi, refreshLlmProviderLabel } = context.ui;
const walk = (root) => [root, ...root.children.flatMap(walk)];
const find = (root, predicate) => walk(root).find(predicate);
const button = (root, text) => find(root, (item) => item.tagName === "button" && item.textContent === text);
const field = (root, name) => find(root, (item) => item.name === name);
const status = (root) => find(root, (item) => item.attributes.role === "status").textContent;
const widget = (node, name) => node.widgets.find((item) => item.name === name);
const tick = async () => { for (let i = 0; i < 8; i++) await Promise.resolve(); };
const ok = (data) => ({ ok: true, json: async () => ({ success: true, ...data }) });
function node() {
    return {
        widgets: [
            { name: "base_url", value: "https://legacy.invalid/v1", inputEl: { style: {}, value: "https://legacy.invalid/v1" } },
            { name: "api_key", value: "legacy-secret", inputEl: { style: {}, value: "legacy-secret" } },
            { name: "source_image_hash", value: "old-hash" },
            { name: "refresh_models", type: "button" },
        ],
        properties: {},
        addDOMWidget(name, type, element, options) {
            const result = { name, type, element, options };
            this.widgets.push(result);
            return result;
        },
        setDirtyCanvas() {},
    };
}
async function open(current, providers) {
    fetchHandler = async () => ok({ providers });
    button(current.widgets.at(-1).element, "选择提供商…").onclick();
    await tick();
    return document.body.children.at(-1);
}
async function main() {
    const current = node();
    const originals = [...current.widgets];
    let switches = 0;
    installLlmProviderUi(current, async () => { switches++; });
    installLlmProviderUi(current);
    assert.equal(current.widgets.length, originals.length + 1);
    assert.deepEqual(current.widgets.slice(0, -1), originals);
    assert.equal(current.widgets.at(-1).options.serialize, false);
    assert.equal(widget(current, "base_url").hidden, true);
    assert.equal(widget(current, "api_key").inputEl.style.display, "none");
    const label = current.widgets.at(-1).element.children[0];
    assert.match(label.textContent, /旧配置/);
    const provider = { id: "saved-id", name: "<img src=x onerror=alert(1)>", base_url: "https://saved.invalid/v1", has_key: true };
    let dialog = await open(current, [provider]);
    assert.equal(requests.at(-1).url, "/aerince/llm/providers");
    assert.equal(requests.at(-1).options.method, "GET");
    assert.equal(field(dialog, "base_url").value, "https://legacy.invalid/v1");
    assert.equal(field(dialog, "api_key").value, "legacy-secret");
    assert.equal(document.activeElement, field(dialog, "name"));
    button(dialog, provider.name).onclick();
    assert.equal(field(dialog, "api_key").value, "");
    assert.match(field(dialog, "api_key").placeholder, /留空保留/);
    current.aerinceReverseInFlight = true;
    await button(dialog, "使用选中").onclick();
    assert.match(status(dialog), /反推进行中/);
    assert.equal(widget(current, "api_key").value, "legacy-secret");
    assert.equal(switches, 0);
    current.aerinceReverseInFlight = false;
    await button(dialog, "使用选中").onclick();
    assert.equal(widget(current, "base_url").value, "aerince-provider://saved-id");
    assert.equal(widget(current, "api_key").value, "");
    assert.equal(widget(current, "api_key").inputEl.value, "");
    assert.equal(widget(current, "source_image_hash").value, "");
    assert.equal(label.textContent, provider.name);
    assert.equal(dialog.open, false);
    assert.equal(switches, 1);

    dialog = await open(current, [provider]);
    assert.equal(field(dialog, "api_key").value, "");
    let finishSave;
    fetchHandler = async (url, options) => {
        assert.equal(url, "/aerince/llm/providers/save");
        const body = JSON.parse(options.body);
        assert.equal(body.id, provider.id);
        assert.equal("api_key" in body, false, "blank edit preserves stored key");
        return new Promise((resolve) => { finishSave = resolve; });
    };
    const beforeSave = requests.length;
    const saving = button(dialog, "保存").onclick();
    await button(dialog, "保存").onclick();
    assert.equal(requests.length, beforeSave + 1, "duplicate save blocked");
    finishSave(ok({ provider }));
    await saving;
    field(dialog, "clear_key").checked = true;
    fetchHandler = async (url, options) => {
        assert.equal(JSON.parse(options.body).api_key, "");
        return ok({ provider: { ...provider, has_key: false } });
    };
    await button(dialog, "保存").onclick();
    button(dialog, "新增").onclick();
    field(dialog, "name").value = "New provider";
    field(dialog, "base_url").value = "https://new.invalid/v1";
    field(dialog, "api_key").value = "new-secret";
    fetchHandler = async (url, options) => {
        assert.deepEqual(JSON.parse(options.body), { name: "New provider", base_url: "https://new.invalid/v1", api_key: "new-secret" });
        return ok({ provider: { id: "new", name: "New provider", base_url: "https://new.invalid/v1", has_key: true } });
    };
    await button(dialog, "保存").onclick();
    assert.equal(widget(current, "base_url").value, "aerince-provider://saved-id", "save does not switch");
    assert.equal(field(dialog, "api_key").value, "");
    const originalValues = current.widgets.map((item) => item.value);
    fetchHandler = async () => ({ ok: false, status: 500, json: async () => ({ success: false, error: "save offline" }) });
    await button(dialog, "保存").onclick();
    assert.match(status(dialog), /save offline/);
    assert.deepEqual(current.widgets.map((item) => item.value), originalValues);
    for (const code of [404, 405, 200]) {
        fetchHandler = async () => ({ ok: code === 200, status: code, json: async () => {
            if (code !== 200) throw new Error("must not parse unavailable route response");
            throw new SyntaxError("invalid JSON");
        } });
        await button(dialog, "保存").onclick();
        assert.ok(status(dialog).includes(`HTTP ${code}`));
        assert.match(status(dialog), code === 200 ? /非 JSON/ : /重启 ComfyUI/);
        assert.deepEqual(current.widgets.map((item) => item.value), originalValues);
    }
    button(dialog, "关闭").onclick();

    fetchHandler = async () => { throw new Error("list offline"); };
    button(current.widgets.at(-1).element, "选择提供商…").onclick();
    await tick();
    dialog = document.body.children.at(-1);
    assert.match(status(dialog), /list offline/);
    assert.deepEqual(current.widgets.map((item) => item.value), originalValues);
    button(dialog, "关闭").onclick();

    const requestCount = requests.length;
    current.properties.aerinceLlmProvider = { id: "restored", name: "Restored name" };
    widget(current, "base_url").value = "aerince-provider://restored";
    refreshLlmProviderLabel(current);
    assert.equal(label.textContent, "Restored name");
    widget(current, "base_url").value = "aerince-provider://another";
    refreshLlmProviderLabel(current);
    assert.equal(label.textContent, "another", "reference wins over stale metadata");
    widget(current, "base_url").value = "https://legacy.invalid/v1";
    refreshLlmProviderLabel(current);
    assert.match(label.textContent, /旧配置/);
    assert.equal(requests.length, requestCount, "configure label restoration never reads secrets");
    const mock = node();
    delete mock.addDOMWidget;
    installLlmProviderUi(mock);
    assert.equal(mock.widgets.length, 4);
    assert.equal(widget(mock, "api_key").hidden, true);
    console.log("PASS: appended nonserialized UI; list/save contract; key omission/clear; duplicate save; safe labels; switch guard/reference/hash; errors preserve state; offline restore");
}
main().catch((error) => { console.error(error); process.exitCode = 1; });
