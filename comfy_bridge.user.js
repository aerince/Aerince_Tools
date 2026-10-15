// ==UserScript==
// @name         ComfyUI Bridge
// @namespace    comfy-bridge
// @version      1.0
// @description  鼠标悬停图片 → 浮窗按钮 → 一键发送到 ComfyUI BridgeReceiver 节点
// @author       comfy-bridge
// @match        *://*/*
// @grant        GM_xmlhttpRequest
// @grant        GM_setValue
// @grant        GM_getValue
// @grant        GM_registerMenuCommand
// @run-at       document-end
// ==/UserScript==

(function () {
    'use strict';

    // ═══════════════════════════════════════════
    //  配置
    // ═══════════════════════════════════════════
    const MIN_SIZE = 200;          // 最小图片尺寸（小于此值不显示浮窗）
    const DEBOUNCE_MS = 100;       // 悬停防抖
    const DEFAULT_COMFY = 'http://127.0.0.1:18901';

    // ═══════════════════════════════════════════
    //  状态
    // ═══════════════════════════════════════════
    let overlay = null;
    let currentImg = null;
    let hoverTimer = null;
    let comfyUrl = GM_getValue('comfy_url', DEFAULT_COMFY);

    // ═══════════════════════════════════════════
    //  CSS 注入（只注入一次）
    // ═══════════════════════════════════════════
    if (!document.getElementById('cb-bridge-style')) {
        const style = document.createElement('style');
        style.id = 'cb-bridge-style';
        style.textContent = `
            @keyframes cb-slide-in  { from{transform:translateX(80px);opacity:0} to{transform:translateX(0);opacity:1} }
            @keyframes cb-slide-out { from{transform:translateX(0);opacity:1} to{transform:translateX(80px);opacity:0} }
            .cb-btn {
                padding:7px 14px; border:none; border-radius:7px; color:#fff; cursor:pointer;
                font-size:12px; font-weight:700; white-space:nowrap; letter-spacing:.3px;
                transition: all .15s; outline:none; font-family:'Segoe UI',system-ui,sans-serif;
            }
            .cb-btn:hover  { filter:brightness(1.25); transform:scale(1.03); }
            .cb-btn:active { transform:scale(0.96); }
        `;
        document.head.appendChild(style);
    }

    // ═══════════════════════════════════════════
    //  浮窗组件
    // ═══════════════════════════════════════════
    function ensureOverlay() {
        if (overlay) return;
        overlay = document.createElement('div');
        overlay.id = 'cb-overlay';
        overlay.style.cssText = `
            position:fixed; display:none; z-index:2147483647;
            background:rgba(18,18,32,.94); border-radius:9px; gap:6px;
            padding:6px 8px; align-items:center; box-shadow:0 4px 20px rgba(0,0,0,.55);
            pointer-events:auto; user-select:none; border:1px solid rgba(255,255,255,.08);
        `;

        // 发送按钮
        const btnSend = document.createElement('button');
        btnSend.className = 'cb-btn';
        btnSend.textContent = '📤 发送';
        btnSend.style.background = 'rgba(123,104,238,.88)';
        btnSend.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); doSend(false); });

        // 发送并运行按钮
        const btnRun = document.createElement('button');
        btnRun.className = 'cb-btn';
        btnRun.textContent = '⚡ 发送并运行';
        btnRun.style.background = 'rgba(0,196,120,.88)';
        btnRun.addEventListener('click', e => { e.preventDefault(); e.stopPropagation(); doSend(true); });

        overlay.appendChild(btnSend);
        overlay.appendChild(btnRun);
        document.body.appendChild(overlay);
    }

    function showOverlay(img) {
        ensureOverlay();
        const r = img.getBoundingClientRect();
        // 计算位置：尽量放在图片右上角，不出屏幕
        let left = r.right - overlay.offsetWidth - 6;
        let top = r.top + 6;
        if (left < 4) left = r.left + 4;
        if (top + overlay.offsetHeight > window.innerHeight - 4) {
            top = r.bottom - overlay.offsetHeight - 6;
        }
        if (top < 4) top = r.bottom + 6;
        overlay.style.left = left + 'px';
        overlay.style.top = top + 'px';
        overlay.style.display = 'flex';
    }

    function hideOverlay() {
        if (overlay) overlay.style.display = 'none';
        currentImg = null;
        if (hoverTimer) { clearTimeout(hoverTimer); hoverTimer = null; }
    }

    // ═══════════════════════════════════════════
    //  Toast 通知
    // ═══════════════════════════════════════════
    function toast(msg, color) {
        const old = document.getElementById('cb-toast');
        if (old) { old.style.animation = 'cb-slide-out .3s ease forwards'; setTimeout(() => old.remove(), 300); }

        const t = document.createElement('div');
        t.id = 'cb-toast';
        t.textContent = msg;
        t.style.cssText = `
            position:fixed; bottom:24px; right:24px; padding:11px 22px; z-index:2147483647;
            background:#111827; color:${color}; border-radius:9px; font-size:13px; font-weight:500;
            box-shadow:0 4px 18px rgba(0,0,0,.5); border-left:3px solid ${color};
            animation:cb-slide-in .3s ease; pointer-events:none;
            font-family:'Segoe UI',system-ui,sans-serif;
        `;
        document.body.appendChild(t);
        setTimeout(() => {
            t.style.animation = 'cb-slide-out .28s ease forwards';
            setTimeout(() => { if (t.parentNode) t.remove(); }, 300);
        }, 2400);
    }

    // ═══════════════════════════════════════════
    //  核心：抓图 + 发送
    // ═══════════════════════════════════════════
    function doSend(trigger) {
        if (!currentImg) return;
        const src = currentImg.src;
        if (!src || src.startsWith('blob:')) {
            toast('⚠ 无法获取图片源', '#fbbf24');
            return;
        }

        toast(trigger ? '⚡ 发送并运行...' : '📤 发送中...', '#a78bfa');

        // 用 GM_xmlhttpRequest 抓图（携带页面 Cookie/Referer，能过认证）
        GM_xmlhttpRequest({
            method: 'GET',
            url: src,
            responseType: 'arraybuffer',
            headers: { 'Referer': window.location.origin + '/' },
            timeout: 15000,
            onload: function (resp) {
                if (resp.status === 200 && resp.response) {
                    const bytes = new Uint8Array(resp.response);
                    const b64 = arrayBufferToBase64(bytes);
                    postToComfy(b64, src, trigger);
                } else {
                    // 降级：只发 URL 让 ComfyUI 自己下载
                    toast('抓取失败，尝试 URL 模式...', '#fbbf24');
                    postToComfy(null, src, trigger);
                }
            },
            onerror: function () {
                toast('抓取失败，尝试 URL 模式...', '#fbbf24');
                postToComfy(null, src, trigger);
            },
            ontimeout: function () {
                toast('抓取超时，尝试 URL 模式...', '#fbbf24');
                postToComfy(null, src, trigger);
            },
        });
    }

    /** Uint8Array → base64（分块处理大图） */
    function arrayBufferToBase64(bytes) {
        const chunkSize = 0x8000;
        const chunks = [];
        for (let i = 0; i < bytes.length; i += chunkSize) {
            const chunk = bytes.subarray(i, Math.min(i + chunkSize, bytes.length));
            chunks.push(String.fromCharCode.apply(null, chunk));
        }
        return btoa(chunks.join(''));
    }

    function postToComfy(b64, url, trigger) {
        const payload = { url: url, trigger: trigger };
        if (b64) payload.data = b64;

        GM_xmlhttpRequest({
            method: 'POST',
            url: comfyUrl + '/bridge/receive',
            headers: { 'Content-Type': 'application/json' },
            data: JSON.stringify(payload),
            timeout: 30000,
            onload: function (resp) {
                try {
                    const r = JSON.parse(resp.responseText);
                    if (r.success) {
                        if (r.trigger_ok === true) {
                            toast('✓ 已接收 — 已触发运行', '#34d399');
                        } else if (r.trigger_ok === false) {
                            toast('⚠ 已接收 — 但运行触发失败，请手动点 Queue', '#fbbf24');
                        } else {
                            toast('✓ 已接收' + (trigger ? '' : ''), '#34d399');
                        }
                    } else {
                        toast('✗ ' + (r.error || '未知错误'), '#f87171');
                    }
                } catch (e) {
                    toast('✗ 无法连接 ComfyUI，服务在运行吗？', '#f87171');
                }
            },
            onerror: function () {
                toast('✗ 无法连接 ComfyUI (' + comfyUrl + ')', '#f87171');
            },
        });
    }

    // ═══════════════════════════════════════════
    //  鼠标悬停检测（修复版：定时器不打架）
    // ═══════════════════════════════════════════
    let hideTimer = null;   // 隐藏延迟定时器

    function cancelHide() {
        if (hideTimer) { clearTimeout(hideTimer); hideTimer = null; }
    }

    function scheduleHide(delay) {
        cancelHide();
        hideTimer = setTimeout(() => {
            hideOverlay();
            hideTimer = null;
        }, delay);
    }

    document.addEventListener('mouseover', function (e) {
        // 1) 鼠标在浮窗上 → 取消隐藏
        if (overlay && overlay.contains(e.target)) {
            cancelHide();
            return;
        }

        // 2) 鼠标在图片上
        const img = e.target.closest('img');
        if (!img) {
            // 鼠标不在任何图片上 → 延迟隐藏
            if (currentImg && !overlay?.contains(e.relatedTarget)) {
                scheduleHide(400);
            }
            return;
        }

        // 过滤过小图片
        if (img.naturalWidth < MIN_SIZE || img.naturalHeight < MIN_SIZE) return;
        const r = img.getBoundingClientRect();
        if (r.width < 50 || r.height < 50) return;

        // 忽略 SVG / 占位图
        const src = img.src || '';
        if (src.includes('data:image/svg')) return;

        // 同一个图片 → 保持浮窗，取消隐藏
        if (img === currentImg) {
            cancelHide();
            return;
        }

        // 新图片 → 取消旧定时器，展示浮窗
        cancelHide();
        if (hoverTimer) { clearTimeout(hoverTimer); }
        hoverTimer = setTimeout(() => {
            currentImg = img;
            showOverlay(img);
        }, DEBOUNCE_MS);
    }, true);

    document.addEventListener('mouseout', function (e) {
        // 鼠标离开图片
        const img = e.target.closest('img');
        if (img === currentImg) {
            if (hoverTimer) { clearTimeout(hoverTimer); hoverTimer = null; }
            // 如果鼠标是去了浮窗，浮窗自己的 mouseover 会 cancelHide
            // 否则延迟隐藏
            scheduleHide(500);
        }
        // 鼠标离开浮窗
        if (overlay && overlay.contains(e.target) && !overlay.contains(e.relatedTarget)) {
            scheduleHide(400);
        }
    }, true);

    // ═══════════════════════════════════════════
    //  键盘快捷键：Ctrl+Shift+S = 仅发送
    //             Ctrl+Shift+X = 发送并运行
    // ═══════════════════════════════════════════
    document.addEventListener('keydown', function (e) {
        if (!e.ctrlKey || !e.shiftKey) return;
        if (e.key === 'S' || e.key === 's') {
            e.preventDefault();
            if (currentImg) doSend(false);
            else {
                // 尝试找页面上最大的图片
                currentImg = findLargestImage();
                if (currentImg) { doSend(false); }
                else toast('⚠ 页面未检测到合适图片，请先悬停到图片上', '#fbbf24');
            }
        } else if (e.key === 'X' || e.key === 'x') {
            e.preventDefault();
            if (currentImg) doSend(true);
            else {
                currentImg = findLargestImage();
                if (currentImg) { doSend(true); }
                else toast('⚠ 页面未检测到合适图片', '#fbbf24');
            }
        }
    });

    function findLargestImage() {
        const imgs = document.querySelectorAll('img');
        let best = null, bestArea = 0;
        for (const img of imgs) {
            if (img.naturalWidth < MIN_SIZE || img.naturalHeight < MIN_SIZE) continue;
            const r = img.getBoundingClientRect();
            if (r.bottom < 0 || r.top > window.innerHeight) continue;
            const area = r.width * r.height;
            if (area > bestArea) { bestArea = area; best = img; }
        }
        return best;
    }

    // ═══════════════════════════════════════════
    //  启动
    // ═══════════════════════════════════════════
    console.log('%c[ComfyUI Bridge] %cReady',
        'color:#a78bfa;font-weight:bold', 'color:#e5e7eb');
    console.log('  目标: ' + comfyUrl + '/bridge/receive');
    console.log('  悬停图片 → 浮窗发送');
    console.log('  快捷键: Ctrl+Shift+S (发送)  Ctrl+Shift+X (发送并运行)');
    console.log('  修改目标 → Tampermonkey 菜单 → ⚙ Set ComfyUI URL');

    // Tampermonkey 菜单命令
    if (typeof GM_registerMenuCommand === 'function') {
        GM_registerMenuCommand('⚙ 设置 ComfyUI 地址', function () {
            const url = prompt('ComfyUI Bridge 接收地址:\n（默认 http://127.0.0.1:18901）', comfyUrl);
            if (url && url.trim()) {
                comfyUrl = url.trim();
                GM_setValue('comfy_url', comfyUrl);
                toast('✓ 地址已更新: ' + comfyUrl, '#34d399');
            }
        });
        GM_registerMenuCommand('📤 发送最大图片 (Ctrl+Shift+S)', () => {
            currentImg = findLargestImage();
            if (currentImg) doSend(false); else toast('⚠ 未找到图片', '#fbbf24');
        });
        GM_registerMenuCommand('⚡ 发送最大图片并运行 (Ctrl+Shift+X)', () => {
            currentImg = findLargestImage();
            if (currentImg) doSend(true); else toast('⚠ 未找到图片', '#fbbf24');
        });
    }
})();
