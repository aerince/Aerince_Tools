import { api } from "../../scripts/api.js";

const PREFIX = "aerince-provider://";
const states = new WeakMap();
const widget = (node, name) => node.widgets?.find((item) => item.name === name);

function hideCredentials(node) {
    for (const name of ["base_url", "api_key"]) {
        const item = widget(node, name);
        if (!item) continue;
        // Keep the original widgets and serialization positions for old workflows.
        item.hidden = true;
        item.type = "converted-widget";
        item.computeSize = () => [0, -4];
        item.draw = () => {};
        if (item.inputEl?.style) item.inputEl.style.display = "none";
        if (item.element?.style) item.element.style.display = "none";
    }
}

export function refreshLlmProviderLabel(node) {
    hideCredentials(node);
    const base = String(widget(node, "base_url")?.value || "");
    const id = base.startsWith(PREFIX) ? base.slice(PREFIX.length) : "";
    const saved = node.properties?.aerinceLlmProvider;
    const label = id ? (saved?.id === id && saved.name ? saved.name : id) : "旧配置 / 未保存提供商";
    const state = states.get(node);
    if (state) {
        state.label.textContent = label;
        state.label.title = id ? `提供商：${label}` : base;
    }
    node.setDirtyCanvas?.(true, true);
}

async function request(path, body) {
    const response = await api.fetchApi(`/aerince/llm/providers${path}`, body === undefined ? { method: "GET" } : {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    if (response.status === 404 || response.status === 405) {
        throw new Error(`提供商接口不可用（HTTP ${response.status}）。请重启 ComfyUI 后按 Ctrl+F5 刷新页面；若仍报错，请检查 Aerince_Tools 的启动日志。`);
    }
    let data;
    try { data = await response.json(); }
    catch { throw new Error(`提供商接口返回了非 JSON 响应（HTTP ${response.status}），请检查 ComfyUI 服务及接口地址。`); }
    if (!response.ok || !data?.success) throw new Error(data?.error || `HTTP ${response.status}`);
    return data;
}

function element(tag, text, parent) {
    const item = document.createElement(tag);
    if (text) item.textContent = text;
    parent?.append(item);
    return item;
}

function openManager(node, state) {
    if (state.dialog?.open) return;
    const dialog = element("dialog");
    state.dialog = dialog;
    dialog.className = "aerince-provider-dialog";
    dialog.setAttribute("aria-label", "LLM 提供商管理");
    const style = element("style", "", dialog);
    style.textContent = `
        .aerince-provider-dialog { color:var(--input-text,#ddd); background:var(--comfy-menu-bg,#222); border:1px solid var(--border-color,#555); border-radius:10px; width:min(740px,90vw); padding:20px; font:14px sans-serif; }
        .aerince-provider-dialog::backdrop { background:#0009; }
        .aerince-provider-dialog h2 { margin:0 0 16px; font-size:18px; }
        .aerince-provider-dialog .provider-columns { display:grid; grid-template-columns:minmax(150px,1fr) minmax(0,2fr); gap:18px; }
        .aerince-provider-dialog nav { display:flex; flex-direction:column; gap:6px; max-height:340px; overflow:auto; }
        .aerince-provider-dialog button,.aerince-provider-dialog input { font:inherit; color:inherit; background:var(--comfy-input-bg,#333); border:1px solid var(--border-color,#666); border-radius:5px; padding:8px; }
        .aerince-provider-dialog button { cursor:pointer; overflow-wrap:anywhere; }
        .aerince-provider-dialog button:disabled { opacity:.5; cursor:default; }
        .aerince-provider-dialog button[aria-pressed=true] { border-color:var(--p-button-primary-background,#77aaff); }
        .aerince-provider-dialog label { display:block; margin:0 0 12px; }
        .aerince-provider-dialog input:not([type=checkbox]) { display:block; box-sizing:border-box; width:100%; margin-top:5px; }
        .aerince-provider-dialog small { display:block; opacity:.8; line-height:1.5; margin:8px 0; }
        .aerince-provider-dialog footer { display:flex; gap:8px; justify-content:flex-end; flex-wrap:wrap; margin-top:16px; }
        .aerince-provider-dialog [role=status] { white-space:pre-wrap; overflow-wrap:anywhere; min-height:20px; margin-top:12px; }
        @media(max-width:520px) { .aerince-provider-dialog .provider-columns { grid-template-columns:1fr; } .aerince-provider-dialog nav { max-height:120px; } }
    `;
    element("h2", "LLM 提供商", dialog);
    const columns = element("div", "", dialog);
    columns.className = "provider-columns";
    const list = element("nav", "", columns);
    list.setAttribute("aria-label", "已保存的提供商");
    const form = element("form", "", columns);
    const field = (title, name, type = "text") => {
        const label = element("label", title, form);
        const input = element("input", "", label);
        input.name = name;
        input.type = type;
        input.autocomplete = type === "password" ? "new-password" : "off";
        return input;
    };
    const name = field("名称", "name");
    const base = field("API 地址", "base_url", "url");
    const key = field("API Key", "api_key", "password");
    name.required = base.required = true;
    element("small", "编辑已保存的提供商时，Key 留空保留原密钥；密钥不会从服务器读取。新建时可填写当前节点的旧配置。", form);
    const clearKey = field("清除已保存的 Key", "clear_key", "checkbox");
    clearKey.onchange = () => { key.disabled = clearKey.checked; };
    const status = element("div", "", dialog);
    status.setAttribute("role", "status");
    status.setAttribute("aria-live", "polite");
    const footer = element("footer", "", dialog);
    const add = element("button", "新增", footer);
    const save = element("button", "保存", footer);
    const use = element("button", "使用选中", footer);
    const close = element("button", "关闭", footer);
    for (const button of [add, save, use, close]) button.type = "button";
    let providers = [], selected = null, busy = false;
    const setBusy = (value) => {
        busy = value;
        for (const input of [name, base, clearKey, add, save]) input.disabled = value;
        key.disabled = value || clearKey.checked;
        use.disabled = value || !selected;
        for (const button of list.children) button.disabled = value;
    };
    function fill(provider) {
        selected = provider;
        const oldBase = String(widget(node, "base_url")?.value || "");
        name.value = provider?.name || "";
        base.value = provider?.base_url || (!oldBase.startsWith(PREFIX) ? oldBase : "");
        key.value = provider ? "" : (!oldBase.startsWith(PREFIX) ? widget(node, "api_key")?.value || "" : "");
        key.placeholder = provider?.has_key ? "已保存；留空保留" : "未设置 Key";
        clearKey.checked = false;
        setBusy(busy);
        for (const button of list.children) button.setAttribute("aria-pressed", String(button.dataset.providerId === provider?.id));
    }
    function renderList() {
        list.replaceChildren();
        for (const provider of providers) {
            const button = element("button", provider.name, list);
            button.type = "button";
            button.dataset.providerId = provider.id;
            button.onclick = () => { if (!busy) { fill(provider); status.textContent = ""; } };
        }
    }
    add.onclick = () => { if (!busy) { fill(null); status.textContent = ""; name.focus(); } };
    close.onclick = () => dialog.close();
    dialog.addEventListener("close", () => { key.value = ""; dialog.remove(); state.dialog = null; });
    save.onclick = async () => {
        if (busy || !form.reportValidity()) return;
        if (!name.value.trim()) { status.textContent = "请填写名称"; return; }
        const body = { name: name.value.trim(), base_url: base.value.trim() };
        if (selected) body.id = selected.id;
        if (clearKey.checked) body.api_key = "";
        else if (key.value) body.api_key = key.value;
        setBusy(true);
        status.textContent = "保存中…";
        try {
            const { provider } = await request("/save", body);
            if (!provider?.id || !provider.name || !provider.base_url) throw new Error("服务器返回的提供商无效");
            providers = [...providers.filter((item) => item.id !== provider.id), provider];
            renderList();
            fill(provider);
            if (widget(node, "base_url")?.value === PREFIX + provider.id) {
                node.properties ||= {};
                node.properties.aerinceLlmProvider = { id: provider.id, name: provider.name };
                refreshLlmProviderLabel(node);
            }
            status.textContent = "已保存，点击“使用选中”应用到节点。";
        } catch (error) { status.textContent = `保存失败：${error.message || error}`; }
        finally { setBusy(false); }
    };
    form.onsubmit = (event) => { event.preventDefault(); save.onclick(); };
    use.onclick = async () => {
        if (busy || !selected) return;
        if (node.aerinceReverseInFlight) { status.textContent = "手动反推进行中，请完成后再切换提供商。"; return; }
        setBusy(true);
        widget(node, "base_url").value = PREFIX + selected.id;
        widget(node, "api_key").value = "";
        for (const field of ["base_url", "api_key"]) {
            const item = widget(node, field);
            if (item.inputEl) item.inputEl.value = item.value;
        }
        const hash = widget(node, "source_image_hash");
        if (hash) hash.value = "";
        node.properties ||= {};
        node.properties.aerinceLlmProvider = { id: selected.id, name: selected.name };
        refreshLlmProviderLabel(node);
        status.textContent = "正在刷新模型…";
        try { await state.onSwitch?.(); dialog.close(); }
        catch (error) { status.textContent = `提供商已切换，模型刷新失败：${error.message || error}`; }
        finally { setBusy(false); }
    };
    document.body.append(dialog);
    fill(null);
    dialog.showModal();
    name.focus();
    setBusy(true);
    status.textContent = "加载提供商…";
    request("").then((data) => {
        if (!Array.isArray(data.providers)) throw new Error("服务器返回的提供商列表无效");
        providers = data.providers;
        renderList();
        const current = providers.find((item) => PREFIX + item.id === widget(node, "base_url")?.value);
        fill(current || null);
        if (current) {
            node.properties ||= {};
            node.properties.aerinceLlmProvider = { id: current.id, name: current.name };
            refreshLlmProviderLabel(node);
        }
        status.textContent = providers.length ? "" : "暂无提供商，请填写并保存。";
    }).catch((error) => { status.textContent = `加载失败：${error.message || error}`; }).finally(() => setBusy(false));
}

export function installLlmProviderUi(node, onSwitch) {
    hideCredentials(node);
    let state = states.get(node);
    if (state) { state.onSwitch = onSwitch || state.onSwitch; refreshLlmProviderLabel(node); return; }
    if (typeof node.addDOMWidget !== "function" || typeof document === "undefined") return;
    const row = element("div");
    row.style.cssText = "display:flex;align-items:center;gap:8px;width:100%;box-sizing:border-box;padding:4px 8px;color:var(--input-text,#ddd);background:var(--comfy-menu-bg,#222)";
    const label = element("span", "", row);
    label.style.cssText = "flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap";
    const button = element("button", "选择提供商…", row);
    button.type = "button";
    button.style.cssText = "color:inherit;background:var(--comfy-input-bg,#333);border:1px solid var(--border-color,#666);border-radius:4px;padding:5px;cursor:pointer";
    state = { label, onSwitch, dialog: null };
    states.set(node, state);
    button.onclick = () => openManager(node, state);
    const ui = node.addDOMWidget("aerince_provider", "aerince_provider", row, { serialize: false });
    ui.computeSize = () => [0, 36];
    refreshLlmProviderLabel(node);
}
