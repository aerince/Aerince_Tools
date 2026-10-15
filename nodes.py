"""
ComfyUI Bridge Pack
===================
BridgeReceiver  — 浏览器直传 + 本地文件 + 透传 + 内置预览 + 拖放加载
BridgeImageLoader — Universal Image Loader 全格式 + 透传输入

HTTP 服务器监听 127.0.0.1:18901，浏览器油猴脚本 POST 图片到此端点。
"""

import base64
import asyncio
import copy
import hashlib
import io
import json
import os
import re
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from http.client import HTTPException
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, urlopen

import numpy as np
import torch
from PIL import Image, ImageOps
import folder_paths
import comfy.sd
import comfy.utils

from .cl_tagger import AerinceCLTaggerV2
from .dlss5_node import AerinceDLSS5Image
from .llm_providers import list_llm_providers, save_llm_provider, resolve_llm_provider
from .local_llm import AerinceLocalLLMImagePromptReverse, AerinceLocalLLMTextProcess

print("[BridgePack] loading ...")

SUPPORTED_FORMATS = ['.png', '.jpg', '.jpeg', '.webp', '.bmp', '.tiff', '.tif', '.gif', '.avif', '.heic', '.heif']

BRIDGE_PORT = int(os.environ.get("COMFY_BRIDGE_PORT", "18901"))
MAX_BODY_SIZE = 100 * 1024 * 1024
MAX_RECEIVER_IMAGE_SIZE = 8 * 1024 * 1024
PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
LORA_RECIPE_DATA_DIR = os.path.join(PLUGIN_DIR, "user_data", "lora_recipes")
LORA_RECIPE_COVER_DIR = os.path.join(LORA_RECIPE_DATA_DIR, "covers")
LORA_RECIPE_FILE = os.path.join(LORA_RECIPE_DATA_DIR, "recipes.json")


class AnyType(str):
    def __ne__(self, __value):
        return False


any_type = AnyType("*")
THINKING_STRENGTH_OPTIONS = ("auto", "off", "low", "medium", "high")
THINKING_MODEL_HINTS = (
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
)
CIVITAI_MODEL_VERSION_BY_HASH_CACHE = {}
CIVITAI_COVER_MISS_CACHE = set()
CIVITAI_FILE_HASH_CACHE = {}
CIVITAI_COVER_SEMAPHORE = threading.Semaphore(2)
COMMON_NON_TRIGGER_TAGS = {
    "1girl",
    "1boy",
    "2girls",
    "2boys",
    "solo",
    "multiple girls",
    "multiple boys",
    "girl",
    "boy",
    "female",
    "male",
    "woman",
    "man",
    "person",
    "people",
    "looking at viewer",
    "looking away",
    "looking back",
    "smile",
    "smiling",
    "open mouth",
    "closed mouth",
    "blush",
    "standing",
    "sitting",
    "lying",
    "kneeling",
    "upper body",
    "full body",
    "cowboy shot",
    "portrait",
    "close-up",
    "close up",
    "face",
    "hair",
    "long hair",
    "short hair",
    "medium hair",
    "black hair",
    "brown hair",
    "blonde hair",
    "white hair",
    "blue hair",
    "red hair",
    "pink hair",
    "purple hair",
    "green hair",
    "eyes",
    "blue eyes",
    "brown eyes",
    "green eyes",
    "red eyes",
    "yellow eyes",
    "purple eyes",
    "closed eyes",
    "shirt",
    "dress",
    "skirt",
    "pants",
    "jacket",
    "shoes",
    "socks",
    "background",
    "simple background",
    "white background",
    "black background",
    "transparent background",
    "indoors",
    "outdoors",
    "day",
    "night",
    "sky",
    "cloud",
    "tree",
    "highres",
    "masterpiece",
    "best quality",
    "good quality",
    "absurdres",
    "official art",
    "very aesthetic",
}


# ═══════════════════════════════════════════════
#  全局桥接状态
# ═══════════════════════════════════════════════
class _BridgeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.queue = deque()        # 每条: (image_tensor, mask_tensor, url, uid)
        self.counter = 0            # 单调递增，每个到达的图片分配唯一 uid
        self.last_consumed = 0      # 上一次 IS_CHANGED 消费时返回的 uid
        self.last_trigger_uid = 0   # 最后一次带有 trigger=true 的图片 uid（供前端轮询）

        self.cached_prompt = None
        self.cached_workflow = None
        self.cached_client_id = None
        self.cached_at = 0.0
        self.last_queued_prompt_id = ""
        self.last_direct_queued_uid = 0

    def push(self, img_t, mask_t, url):
        """HTTP handler 线程调用：存入队列并返回分配的 uid"""
        with self.lock:
            self.counter += 1
            uid = self.counter
            self.queue.append((img_t.clone(), mask_t.clone() if mask_t is not None else torch.zeros((1, img_t.shape[1], img_t.shape[2]), dtype=torch.float32), url, uid))
            return uid

    def peek_uid(self):
        """IS_CHANGED 调用：返回下一个待消费的 uid（不弹出）"""
        with self.lock:
            if self.queue:
                return self.queue[0][3]
            return self.counter

    def pop(self):
        """receive() 调用：弹出队列头部，返回 (img, mask, url) 或 None"""
        with self.lock:
            if self.queue:
                img, mask, url, uid = self.queue.popleft()
                self.last_consumed = uid
                return img, mask, url
            return None, None, None

    def status(self):
        """返回当前状态快照，供前端轮询"""
        with self.lock:
            return {
                "queue_size": len(self.queue),
                "next_uid": self.queue[0][3] if self.queue else 0,
                "last_consumed": self.last_consumed,
                "last_trigger_uid": self.last_trigger_uid,
                "has_cached_prompt": self.cached_prompt is not None,
                "cached_at": self.cached_at,
                "last_queued_prompt_id": self.last_queued_prompt_id,
                "last_direct_queued_uid": self.last_direct_queued_uid,
            }

    def update_prompt(self, prompt, workflow=None, client_id=None):
        with self.lock:
            self.cached_prompt = copy.deepcopy(prompt)
            self.cached_workflow = copy.deepcopy(workflow)
            self.cached_client_id = client_id
            self.cached_at = time.time()
            return self.cached_at

    def get_prompt_snapshot(self):
        with self.lock:
            if self.cached_prompt is None:
                return None
            return {
                "prompt": copy.deepcopy(self.cached_prompt),
                "workflow": copy.deepcopy(self.cached_workflow),
                "client_id": self.cached_client_id,
                "cached_at": self.cached_at,
            }

    def set_last_queued_prompt_id(self, prompt_id, uid=0):
        with self.lock:
            self.last_queued_prompt_id = prompt_id
            if uid:
                self.last_direct_queued_uid = uid

bridge = _BridgeState()


# ═══════════════════════════════════════════════
#  通用图片解码（PIL → tensor + mask）
# ═══════════════════════════════════════════════
def _queue_cached_prompt(uid=0):
    snapshot = bridge.get_prompt_snapshot()
    if not snapshot:
        return False, "No cached workflow. Open or refresh the ComfyUI page once so BridgeReceiver can sync it."

    try:
        import execution
        from server import PromptServer

        ps = PromptServer.instance
        if ps is None:
            return False, "PromptServer is not ready"

        prompt = snapshot["prompt"]
        workflow = snapshot.get("workflow")
        client_id = snapshot.get("client_id") or "comfyui-bridge"

        async def queue_on_comfy_loop():
            prompt_id = str(uuid.uuid4())
            extra_data = {"client_id": client_id}
            if workflow is not None:
                extra_data["extra_pnginfo"] = {"workflow": workflow}

            json_data = {
                "prompt": prompt,
                "client_id": client_id,
                "extra_data": extra_data,
            }
            if hasattr(ps, "trigger_on_prompt"):
                json_data = ps.trigger_on_prompt(json_data)
                queued_prompt = json_data.get("prompt", prompt)
                extra_data = json_data.get("extra_data", extra_data)
            else:
                queued_prompt = prompt

            if hasattr(ps, "node_replace_manager"):
                ps.node_replace_manager.apply_replacements(queued_prompt)

            try:
                validation = execution.validate_prompt(prompt_id, queued_prompt, None)
            except TypeError:
                validation = execution.validate_prompt(prompt_id, queued_prompt)
            if asyncio.iscoroutine(validation):
                validation = await validation

            if not validation[0]:
                return False, f"Prompt validation failed: {validation[1]}"

            outputs_to_execute = validation[2]
            sensitive = {}
            for key in getattr(execution, "SENSITIVE_EXTRA_DATA_KEYS", []):
                if key in extra_data:
                    sensitive[key] = extra_data.pop(key)
            extra_data["create_time"] = int(time.time() * 1000)
            number = -time.time()

            ps.prompt_queue.put((number, prompt_id, queued_prompt, extra_data, outputs_to_execute, sensitive))

            return True, prompt_id

        loop = getattr(ps, "loop", None)
        if loop and loop.is_running():
            future = asyncio.run_coroutine_threadsafe(queue_on_comfy_loop(), loop)
            ok, result = future.result(timeout=15)
        else:
            ok, result = asyncio.run(queue_on_comfy_loop())

        if ok:
            bridge.set_last_queued_prompt_id(result, uid)
        return ok, result
    except Exception as e:
        traceback.print_exc()
        return False, str(e)


def _send_frontend_trigger(uid):
    try:
        from server import PromptServer
        ps = PromptServer.instance
        ps.send_sync("bridge-trigger", {"uid": uid, "timestamp": time.time()})
        return True, ""
    except Exception as e:
        return False, str(e)


def _cache_prompt_from_node(prompt=None, extra_pnginfo=None):
    if not isinstance(prompt, dict):
        return

    workflow = None
    if isinstance(extra_pnginfo, dict):
        workflow = extra_pnginfo.get("workflow")

    cached_at = bridge.update_prompt(prompt, workflow, "comfyui-bridge-node")
    print(f"[Bridge] Cached workflow from node execution at {cached_at:.3f}")


def _pil_to_tensor(img: Image.Image):
    img = ImageOps.exif_transpose(img)

    if img.mode == 'RGBA':
        mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
        img = img.convert('RGB')
    elif img.mode == 'LA':
        mask = np.array(img.getchannel('L')).astype(np.float32) / 255.0
        mask = 1. - torch.from_numpy(mask)
        img = img.convert('RGB')
    elif img.mode == 'P':
        if 'transparency' in img.info:
            img = img.convert('RGBA')
            mask = np.array(img.getchannel('A')).astype(np.float32) / 255.0
            mask = 1. - torch.from_numpy(mask)
            img = img.convert('RGB')
        else:
            img = img.convert('RGB')
            mask = torch.zeros((img.size[1], img.size[0]), dtype=torch.float32)
    elif img.mode == 'L':
        mask = torch.zeros((img.size[1], img.size[0]), dtype=torch.float32)
        img = img.convert('RGB')
    else:
        img = img.convert('RGB')
        mask = torch.zeros((img.size[1], img.size[0]), dtype=torch.float32)

    img_np = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(img_np)[None,], mask.unsqueeze(0)


def _bytes_to_tensor(data: bytes):
    return _pil_to_tensor(Image.open(io.BytesIO(data)))


def _file_to_tensor(path: str):
    return _pil_to_tensor(Image.open(path))


def _tensor_png_size(tensor):
    img_np = (tensor[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, "PNG", optimize=True)
    return buf.tell()


def _compress_tensor_to_byte_limit(image, mask, max_bytes=MAX_RECEIVER_IMAGE_SIZE):
    encoded_size = _tensor_png_size(image)
    if encoded_size <= max_bytes:
        return image, mask, encoded_size, False

    original_height, original_width = image.shape[1:3]
    height, width = original_height, original_width
    while encoded_size > max_bytes and height > 1 and width > 1:
        scale = min(0.95, (max_bytes / encoded_size) ** 0.5 * 0.95)
        height = max(1, int(height * scale))
        width = max(1, int(width * scale))
        image = torch.nn.functional.interpolate(
            image.movedim(-1, 1), size=(height, width), mode="bicubic", align_corners=False
        ).movedim(1, -1).clamp(0, 1)
        encoded_size = _tensor_png_size(image)

    mask = torch.zeros(
        (image.shape[0], image.shape[1], image.shape[2]),
        dtype=torch.float32,
        device=image.device,
    )
    return image, mask, encoded_size, True


def _tensor_to_temp(tensor, filename_prefix="bridge"):
    """保存 tensor 到 ComfyUI temp 目录，返回预览数据结构"""
    temp_dir = folder_paths.get_temp_directory()
    os.makedirs(temp_dir, exist_ok=True)
    filename = f"{filename_prefix}_{int(time.time()*1000):013d}.png"
    filepath = os.path.join(temp_dir, filename)

    img_np = (tensor[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_np).save(filepath, "PNG")

    return {
        "filename": filename,
        "subfolder": "",
        "type": "temp",
    }


def _get_file_list():
    """获取 ComfyUI input 文件夹下所有支持的图片文件"""
    d = folder_paths.get_input_directory()
    files = []
    if os.path.exists(d):
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            if os.path.isfile(fp) and os.path.splitext(f)[1].lower() in SUPPORTED_FORMATS:
                files.append(f)
    return files


def _normalize_openai_base_url(base_url):
    base_url = (base_url or "").strip().rstrip("/")
    if not base_url:
        raise ValueError("base_url is required")
    for suffix in ("/chat/completions", "/models"):
        if base_url.endswith(suffix):
            base_url = base_url[:-len(suffix)].rstrip("/")
    version_suffixes = ("/v1", "/v2", "/v3", "/v4", "/api/v1", "/api/v2", "/api/v3", "/api/v4")
    if not base_url.endswith(version_suffixes):
        base_url = base_url + "/v1"
    return base_url


def _openai_headers(api_key):
    headers = {"Content-Type": "application/json"}
    api_key = (api_key or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _request_json(url, api_key="", method="GET", payload=None, timeout=30):
    body = None
    if payload is not None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = Request(url, data=body, headers=_openai_headers(api_key), method=method)
    try:
        with urlopen(req, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
            return json.loads(text) if text else {}
    except HTTPError as e:
        try:
            detail = e.read().decode("utf-8", errors="replace")
        except Exception:
            detail = str(e)
        raise RuntimeError(f"HTTP {e.code}: {detail}") from e
    except URLError as e:
        raise RuntimeError(f"Request failed: {e}") from e


def _fetch_llm_models(base_url, api_key="", timeout=15):
    base_url, api_key = resolve_llm_provider(base_url, api_key)
    url = _normalize_openai_base_url(base_url) + "/models"
    data = _request_json(url, api_key=api_key, method="GET", timeout=timeout)
    models = []
    model_details = []
    for item in data.get("data", []):
        model_id = item.get("id") if isinstance(item, dict) else str(item)
        if model_id:
            models.append(model_id)
            model_details.append({
                "id": model_id,
                "thinking_capable": _model_supports_thinking(model_id, item if isinstance(item, dict) else {}),
            })
    models = sorted(set(models))
    detail_by_id = {item["id"]: item for item in model_details}
    return models, [detail_by_id[model_id] for model_id in models]


def _model_supports_thinking(model, model_info=None):
    model_text = str(model or "").lower()
    if any(hint in model_text for hint in THINKING_MODEL_HINTS):
        return True
    if not isinstance(model_info, dict):
        return False

    capability_keys = ("capabilities", "features", "supported_features", "metadata")
    values = []
    for key in capability_keys:
        value = model_info.get(key)
        if isinstance(value, dict):
            values.extend(f"{k}:{v}" for k, v in value.items())
        elif isinstance(value, list):
            values.extend(str(item) for item in value)
        elif value is not None:
            values.append(str(value))
    capability_text = " ".join(values).lower()
    return any(word in capability_text for word in ("reasoning", "thinking", "reasoner"))


def _apply_thinking_options(payload, model, thinking_strength="auto"):
    strength = str(thinking_strength or "auto").lower()
    if strength not in THINKING_STRENGTH_OPTIONS:
        strength = "auto"
    if strength == "auto":
        return False

    if strength == "off":
        payload["enable_thinking"] = False
        payload["thinking"] = {"type": "disabled"}
        payload["reasoning"] = {"enabled": False}
        payload["chat_template_kwargs"] = {"enable_thinking": False}
        return True

    effort = "low" if strength == "low" else "medium" if strength == "medium" else "high"
    budget_tokens = {"low": 1024, "medium": 4096, "high": 8192}[effort]
    payload["enable_thinking"] = True
    payload["thinking"] = {"type": "enabled", "budget_tokens": budget_tokens}
    payload["reasoning_effort"] = effort
    payload["chat_template_kwargs"] = {"enable_thinking": True}
    return True


def _request_chat_json(url, api_key, payload, timeout):
    # ponytail: one shared, bounded loop for compatibility and empty-response retries.
    candidate = copy.deepcopy(payload)
    candidate.pop("max_tokens", None)
    candidate["stream"] = True
    empty_retried = False
    deadline = time.monotonic() + float(timeout)
    for attempt in range(1, 4):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("LLM 请求超时，未取得完整提示词。")
        try:
            headers = _openai_headers(api_key)
            headers["Accept"] = "text/event-stream, application/json"
            req = Request(url, data=json.dumps(candidate, ensure_ascii=False).encode("utf-8"),
                          headers=headers, method="POST")
            with urlopen(req, timeout=remaining) as response:
                data = _read_chat_response(response, deadline)
        except HTTPError as error:
            try:
                body = json.loads(error.read(65536).decode("utf-8"))
                detail = body.get("error", body) if isinstance(body, dict) else {}
            except (ValueError, UnicodeError):
                detail = {}
            message = _chat_error_message(detail, api_key)
            keys = ("stream", "enable_thinking", "thinking", "reasoning_effort",
                    "chat_template_kwargs", "reasoning")
            unsupported = re.search(r"unknown|unsupported|not supported|unrecognized|not permitted|extra inputs", message, re.I)
            rejected = [key for key in keys if key in candidate and
                        re.search(r"(?<![\w])" + re.escape(key) + r"(?![\w])", message)]
            if error.code in (400, 422) and unsupported and rejected and attempt < 3:
                changed = copy.deepcopy(candidate)
                for key in rejected:
                    if key == "stream":
                        changed[key] = False
                    else:
                        changed.pop(key)
                requested_thinking = any(key in candidate for key in keys[1:])
                if requested_thinking and not any(key in changed for key in keys[1:]):
                    raise RuntimeError("接口不支持指定的思考控制参数，未静默切回 auto；请检查接口文档。") from None
                if changed != candidate:
                    print(f"[Aerince LLM] 参数回退 attempt={attempt}, rejected={rejected}")
                    candidate = changed
                    continue
            raise RuntimeError(f"LLM HTTP {error.code}: {message}") from None
        except (URLError, OSError, HTTPException, UnicodeError, ValueError) as error:
            # Never include the request/image or a partially received answer in the error.
            raise RuntimeError(f"LLM 连接或响应读取失败（{type(error).__name__}），未采用不完整结果。") from None

        if not isinstance(data, dict):
            raise RuntimeError("LLM 响应格式错误：预期 JSON 对象。")
        if data.get("error"):
            raise RuntimeError("LLM 服务端错误: " + _chat_error_message(data["error"], api_key))
        choice, message = _chat_choice(data)
        if not choice or not isinstance(choice.get("message"), dict):
            raise RuntimeError("LLM 响应格式错误：缺少 choices[0].message，接口可能不兼容 Chat Completions。")
        finish = choice.get("finish_reason")
        text = _extract_chat_text(data)
        summary = _chat_response_summary(data, candidate, attempt)
        if message.get("refusal") or finish in ("content_filter", "safety") or any(
            isinstance(part, dict) and part.get("type") == "refusal"
            for part in (message.get("content") if isinstance(message.get("content"), list) else [])
        ):
            raise RuntimeError(f"模型未提供提示词（拒绝或内容过滤）。{summary}")
        if finish in ("length", "max_tokens"):
            raise RuntimeError(f"服务端输出额度耗尽，结果可能不完整；插件未设置 max_tokens，请检查服务端限制。{summary}")
        if finish not in (None, "stop") or message.get("tool_calls") or message.get("function_call"):
            raise RuntimeError(f"模型未正常结束文本回答。{summary}")
        if text:
            print(f"[Aerince LLM] 完成 {summary}")
            return data
        if not empty_retried and attempt < 3:
            print(f"[Aerince LLM] 正文为空，使用原图和原参数补试一次。{summary}")
            empty_retried = True
            continue
        raise RuntimeError(f"模型最终正文为空，未将思考过程作为提示词。{summary}")


def _chat_error_message(detail, api_key=""):
    if isinstance(detail, dict):
        detail = " ".join(str(detail.get(key) or "") for key in ("message", "param", "code"))
    text = str(detail or "服务端未提供错误详情")
    if api_key:
        text = text.replace(api_key, "[redacted]")
    text = re.sub(r"data:[^\s\"']+", "[image omitted]", text)
    text = re.sub(r"(?i)Bearer\s+\S+", "Bearer [redacted]", text)
    return text[:500]


def _chat_choice(data):
    choices = data.get("choices")
    if not isinstance(choices, list):
        return {}, {}
    choice = next((item for item in choices if isinstance(item, dict) and item.get("index", 0) == 0), {})
    message = choice.get("message")
    return choice, message if isinstance(message, dict) else {}


def _chat_content_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(_chat_content_text(item) for item in value)
    if isinstance(value, dict) and value.get("type", "text") in ("text", "output_text"):
        text = value.get("text", "")
        return text if isinstance(text, str) else ""
    return ""


def _chat_response_summary(data, payload, attempt):
    choice, message = _chat_choice(data)
    reasoning_chars = message.get("_reasoning_chars", 0) + sum(
        len(_chat_content_text(message.get(key))) for key in ("reasoning_content", "reasoning", "thinking", "analysis")
    )
    usage = data.get("usage")
    # Only numeric accounting fields are logged, never arbitrary provider text.
    usage = {key: value for key, value in usage.items() if isinstance(value, (int, float))} if isinstance(usage, dict) else None
    options = {key: payload[key] for key in ("enable_thinking", "thinking", "reasoning_effort", "chat_template_kwargs", "reasoning") if key in payload}
    return json.dumps({"request_model": payload.get("model"), "response_model": data.get("model"),
                       "finish_reason": choice.get("finish_reason"), "usage": usage,
                       "content_chars": len(_chat_content_text(message.get("content"))),
                       "reasoning_chars": reasoning_chars, "thinking_sent": options,
                       "stream": payload.get("stream"), "attempt": attempt}, ensure_ascii=False)


def _read_chat_response(response, deadline):
    first = response.readline()
    while first and not first.strip():
        if time.monotonic() >= deadline:
            raise RuntimeError("LLM 接收超时，未采用不完整结果。")
        first = response.readline()
    content_type = response.headers.get("Content-Type", "").lower()
    if "text/event-stream" not in content_type and not first.lstrip(b"\xef\xbb\xbf").startswith((b"data:", b":", b"event:")):
        raw = first + response.read()
        if time.monotonic() >= deadline:
            raise RuntimeError("LLM 接收超时，未采用不完整结果。")
        return json.loads(raw.decode("utf-8-sig"))

    result = {"choices": [{"index": 0, "message": {"role": "assistant"}, "finish_reason": None}]}
    choice = result["choices"][0]
    message = choice["message"]
    parts, refusals, event_lines = [], [], []
    done = False

    def consume_event():
        nonlocal done
        if not event_lines:
            return
        event = "\n".join(event_lines)
        event_lines.clear()
        if event.strip() == "[DONE]":
            done = True
            return
        chunk = json.loads(event)
        if not isinstance(chunk, dict):
            raise ValueError("Invalid SSE object")
        if chunk.get("error"):
            result["error"] = chunk["error"]
            done = True
            return
        for key in ("model", "usage"):
            if chunk.get(key) is not None:
                result[key] = chunk[key]
        for item in chunk.get("choices") or []:
            if not isinstance(item, dict) or item.get("index", 0) != 0:
                continue
            delta = item.get("delta") or item.get("message") or {}
            if not isinstance(delta, dict):
                raise ValueError("Invalid SSE delta")
            parts.append(_chat_content_text(delta.get("content")))
            message["_reasoning_chars"] = message.get("_reasoning_chars", 0) + sum(
                len(_chat_content_text(delta.get(key))) for key in ("reasoning_content", "reasoning", "thinking", "analysis")
            )
            if delta.get("refusal"):
                refusals.append(str(delta["refusal"]))
            if isinstance(delta.get("content"), list) and any(
                isinstance(part, dict) and part.get("type") == "refusal" for part in delta["content"]
            ):
                refusals.append("refusal")
            for key in ("tool_calls", "function_call"):
                if delta.get(key):
                    message[key] = True
            if item.get("finish_reason") is not None:
                choice["finish_reason"] = item["finish_reason"]

    line = first
    while line:
        if time.monotonic() >= deadline:
            raise RuntimeError("LLM 流式接收超时，未采用不完整结果。")
        decoded = line.decode("utf-8-sig").rstrip("\r\n")
        if not decoded:
            consume_event()
            if done:
                break
        elif decoded.startswith("data:"):
            event_lines.append(decoded[6:] if decoded.startswith("data: ") else decoded[5:])
        line = response.readline()
    consume_event()
    if not done and choice["finish_reason"] is None:
        raise RuntimeError("LLM 流式响应在结束标记前中断，未采用已接收的半截提示词。")
    message["content"] = "".join(parts)
    if refusals:
        message["refusal"] = "".join(refusals)
    return result


def _image_tensor_to_data_url(image):
    if image is None or image.shape[0] == 0:
        raise ValueError("image input is required")
    img_np = (image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil_image = Image.fromarray(img_np)
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _image_tensor_fingerprint(image):
    if image is None or getattr(image, "shape", None) is None or image.shape[0] == 0:
        raise ValueError("image input is required")
    img_np = (image[0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    return _pixel_fingerprint(np.asarray(Image.fromarray(img_np).convert("RGB"), dtype=np.uint8))


def _pixel_fingerprint(img_np):
    # shape + raw pixels: stable, cheap enough for single images
    h = hashlib.sha256()
    h.update(str(tuple(img_np.shape)).encode("utf-8"))
    h.update(img_np.tobytes())
    return h.hexdigest()


def _data_url_fingerprint(data_url):
    data_url = (data_url or "").strip()
    if not data_url:
        raise ValueError("image is required")
    payload = data_url.split(",", 1)[1] if "," in data_url and data_url.startswith("data:") else data_url
    with Image.open(io.BytesIO(base64.b64decode(payload))) as image:
        return _pixel_fingerprint(np.asarray(image.convert("RGB"), dtype=np.uint8))


def _extract_chat_text(data):
    _, message = _chat_choice(data)
    # Reasoning fields are never a reliable source of the final image prompt.
    return _clean_llm_prompt_text(_chat_content_text(message.get("content")))


def _clean_llm_prompt_text(text):
    text = (text or "").strip()
    if not text:
        return ""

    text = re.sub(r"(?is)<think>.*?</think>", "", text)
    text = re.sub(r"(?is)<reasoning>.*?</reasoning>", "", text)
    text = re.sub(r"(?is)<(?:think|reasoning)>.*$", "", text)
    text = re.sub(r"(?is)```(?:think|thinking|reasoning|analysis)\s*.*?```", "", text)
    if re.match(r"(?is)^\s*(?:\*\*)?(?:thinking process|analysis|reasoning)\s*:", text):
        return ""

    if re.match(r"^\s*(?:思考|推理|分析)(?:过程)?\s*[：:]", text):
        return ""
    text = re.sub(r"(?is)^\s*```(?:text|plaintext)?\s*\n(.*?)\n```\s*$", r"\1", text)
    text = re.sub(r"(?i)^\s*(?:(?:final\s+)?prompt|final answer|最终提示词|优化后的提示词|反推提示词)\s*[：:]\s*", "", text)
    return text.strip()


def _llm_reverse_image_prompt_from_data_url(
    data_url,
    base_url,
    api_key,
    model,
    system_prompt,
    user_prompt,
    temperature=0.2,
    timeout=300,
    thinking_strength="auto",
    **legacy_options,
):
    base_url, api_key = resolve_llm_provider(base_url, api_key)
    model = (model or "").strip()
    if not model or model == "refresh_models_first":
        raise ValueError("model is required. Click refresh_models and choose a model.")
    data_url = (data_url or "").strip()
    if not data_url:
        raise ValueError("image is required")
    if not data_url.startswith("data:"):
        data_url = "data:image/png;base64," + data_url
    try:
        header, encoded = data_url.split(",", 1)
        if not header.startswith("data:image/") or not header.endswith(";base64"):
            raise ValueError("Expected a base64 image")
        image_bytes = base64.b64decode(encoded, validate=True)
        with Image.open(io.BytesIO(image_bytes)) as source:
            image_info = {"format": source.format, "width": source.width,
                          "height": source.height, "bytes": len(image_bytes)}
            source.verify()
    except (ValueError, OSError) as error:
        raise ValueError("反推图片无效：需要完整、可解码的 Base64 图片。") from None
    print(f"[Aerince LLM] 图片 {json.dumps(image_info)}")

    url = _normalize_openai_base_url(base_url) + "/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt or ""},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt or "Reverse this image into a prompt."},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            },
        ],
        "temperature": float(temperature),
    }
    _apply_thinking_options(payload, model, thinking_strength)
    data = _request_chat_json(
        url,
        api_key=api_key,
        payload=payload,
        timeout=int(timeout),
    )
    return _extract_chat_text(data)


def _llm_reverse_image_prompt_from_tensor(
    image,
    base_url,
    api_key,
    model,
    system_prompt,
    user_prompt,
    temperature=0.2,
    timeout=300,
    thinking_strength="auto",
    **legacy_options,
):
    return _llm_reverse_image_prompt_from_data_url(
        _image_tensor_to_data_url(image),
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=temperature,
        timeout=timeout,
        thinking_strength=thinking_strength,
    )


def _ensure_lora_recipe_dirs():
    os.makedirs(LORA_RECIPE_COVER_DIR, exist_ok=True)



def _empty_lora_recipe_store():
    return {"version": 1, "recipes": []}


def _read_lora_recipes():
    _ensure_lora_recipe_dirs()
    if not os.path.exists(LORA_RECIPE_FILE):
        return _empty_lora_recipe_store()
    try:
        with open(LORA_RECIPE_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
    except Exception as exc:
        print(f"[AerinceTools] failed to read lora recipes: {exc}")
        raise ValueError(f"LoRA recipe file is unreadable: {exc}") from exc
    if not isinstance(data, dict):
        return _empty_lora_recipe_store()
    recipes = data.get("recipes")
    if not isinstance(recipes, list):
        data["recipes"] = []
    data["version"] = int(data.get("version") or 1)
    return data


def _write_lora_recipes(data):
    _ensure_lora_recipe_dirs()
    payload = {
        "version": int(data.get("version") or 1) if isinstance(data, dict) else 1,
        "recipes": data.get("recipes", []) if isinstance(data, dict) else [],
    }
    tmp_path = LORA_RECIPE_FILE + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, LORA_RECIPE_FILE)
    return payload


def _sanitize_recipe_id(value):
    text = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(value or "").strip()).strip("_")
    return text or f"recipe_{uuid.uuid4().hex[:10]}"


def _normalize_recipe_loras(loras):
    normalized = []
    for item in loras or []:
        if not isinstance(item, dict):
            continue
        name = _normalize_lora_name(item.get("name") or item.get("lora") or "")
        if not name:
            continue
        try:
            strength = float(item.get("strength", 1.0))
        except Exception:
            strength = 1.0
        normalized.append({
            "enabled": bool(item.get("enabled", True)),
            "name": name,
            "strength": strength,
        })
    return normalized


def _normalize_lora_recipe(recipe, existing_id=None):
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    if not isinstance(recipe, dict):
        recipe = {}
    recipe_id = _sanitize_recipe_id(recipe.get("id") or existing_id or f"recipe_{uuid.uuid4().hex}")
    name = str(recipe.get("name") or recipe_id).strip() or recipe_id
    created_at = str(recipe.get("created_at") or now)
    return {
        "id": recipe_id,
        "name": name,
        "description": str(recipe.get("description") or ""),
        "cover": str(recipe.get("cover") or ""),
        "model_tags": [str(x) for x in recipe.get("model_tags", []) if str(x).strip()] if isinstance(recipe.get("model_tags"), list) else [],
        "trigger_words": str(recipe.get("trigger_words") or ""),
        "loras": _normalize_recipe_loras(recipe.get("loras")),
        "created_at": created_at,
        "updated_at": now,
    }


def _recipe_cover_url(relative_cover):
    relative_cover = str(relative_cover or "").replace("\\", "/").lstrip("/")
    if not relative_cover:
        return ""
    return f"/aerince/lora-recipes/cover?path={relative_cover}"


def _public_lora_recipe(recipe):
    item = dict(recipe)
    item["cover_url"] = _recipe_cover_url(item.get("cover"))
    return item


def _extract_trigger_words_for_loras(loras, max_words_per_lora=1, separator=", ", use_civitai_fallback=True, civitai_timeout=10, filter_common_tags=True, extra_excluded_tags=""):
    all_words = []
    details = []
    missing = []
    for lora in _normalize_recipe_loras(loras):
        if not lora.get("enabled", True):
            continue
        name = lora.get("name", "")
        path = folder_paths.get_full_path("loras", name)
        if not path or not os.path.exists(path):
            missing.append(name)
            details.append({"lora": name, "trigger_words": [], "trigger_source": "missing_file", "missing": True})
            continue
        words, metadata, lookup = _trigger_words_for_lora(
            name,
            int(max_words_per_lora),
            bool(use_civitai_fallback),
            int(civitai_timeout),
            bool(filter_common_tags),
            extra_excluded_tags,
        )
        all_words.extend(words)
        details.append({
            "lora": name,
            "strength": lora.get("strength", 1.0),
            "trigger_words": words,
            "trigger_source": lookup.get("source"),
            "filtered_words": lookup.get("filtered_words"),
            "metadata_keys": sorted(metadata.keys()) if isinstance(metadata, dict) else [],
        })
    return _format_words(_dedupe_words(all_words), separator), details, missing


def _safe_cover_path(relative_path):
    relative_path = str(relative_path or "").replace("\\", "/").lstrip("/")
    if not relative_path.startswith("covers/"):
        relative_path = "covers/" + os.path.basename(relative_path)
    full = os.path.abspath(os.path.join(LORA_RECIPE_DATA_DIR, relative_path))
    base = os.path.abspath(LORA_RECIPE_COVER_DIR)
    if not full.startswith(base + os.sep) and full != base:
        raise ValueError("Invalid cover path")
    return relative_path, full


def _register_aerince_routes():
    try:
        from aiohttp import web
        from server import PromptServer

        routes = PromptServer.instance.routes

        @routes.get("/aerince/llm/providers")
        async def aerince_llm_providers(request):
            try:
                providers = await asyncio.to_thread(list_llm_providers)
                return web.json_response({"success": True, "providers": providers})
            except (ValueError, OSError):
                return web.json_response({"success": False, "error": "无法读取提供商配置，请检查本地文件格式与权限。"}, status=400)

        @routes.post("/aerince/llm/providers/save")
        async def aerince_llm_provider_save(request):
            try:
                data = await request.json()
                provider = await asyncio.to_thread(save_llm_provider, data)
                return web.json_response({"success": True, "provider": provider})
            except ValueError as error:
                return web.json_response({"success": False, "error": str(error)}, status=400)
            except OSError:
                return web.json_response({"success": False, "error": "保存提供商失败，请检查本地目录的写入权限。"}, status=500)

        @routes.post("/aerince/llm/models")
        async def aerince_llm_models(request):
            try:
                data = await request.json()
                models, model_details = await asyncio.to_thread(
                    _fetch_llm_models,
                    data.get("base_url", ""),
                    data.get("api_key", ""),
                    timeout=15,
                )
                return web.json_response({"success": True, "models": models, "model_details": model_details})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=400)

        @routes.post("/aerince/llm/reverse-image")
        async def aerince_llm_reverse_image(request):
            try:
                data = await request.json()
                image_data_url = data.get("image_data_url") or data.get("image") or ""
                image_hash = await asyncio.to_thread(_data_url_fingerprint, image_data_url)
                text = await asyncio.to_thread(
                    _llm_reverse_image_prompt_from_data_url,
                    image_data_url,
                    base_url=data.get("base_url", ""),
                    api_key=data.get("api_key", ""),
                    model=data.get("model", ""),
                    system_prompt=data.get("system_prompt", ""),
                    user_prompt=data.get("user_prompt", ""),
                    temperature=data.get("temperature", 0.2),
                    timeout=data.get("timeout", 300),
                    thinking_strength=data.get("thinking_strength", "auto"),
                )
                return web.json_response({"success": True, "prompt": text, "image_hash": image_hash})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=400)

        @routes.get("/aerince/lora-recipes")
        async def aerince_lora_recipes_list(request):
            data = _read_lora_recipes()
            return web.json_response({"success": True, "recipes": [_public_lora_recipe(r) for r in data.get("recipes", [])]})

        @routes.post("/aerince/lora-recipes/save")
        async def aerince_lora_recipes_save(request):
            try:
                data = await request.json()
                recipe = _normalize_lora_recipe(data.get("recipe", data))
                overwrite = bool(data.get("overwrite", True))
                store = _read_lora_recipes()
                recipes = store.get("recipes", [])
                existing_index = next((i for i, item in enumerate(recipes) if item.get("id") == recipe["id"]), -1)
                if existing_index >= 0:
                    if not overwrite:
                        return web.json_response({"success": False, "error": "recipe_exists", "id": recipe["id"]}, status=409)
                    existing = recipes[existing_index]
                    recipe["created_at"] = existing.get("created_at", recipe["created_at"])
                    if not recipe["cover"]:
                        # The save dialog has no way to clear a cover, so an empty cover
                        # on overwrite always means "keep the original cover".
                        recipe["cover"] = str(existing.get("cover") or "")
                    recipes[existing_index] = recipe
                else:
                    recipes.append(recipe)
                store["recipes"] = recipes
                _write_lora_recipes(store)
                return web.json_response({"success": True, "recipe": _public_lora_recipe(recipe)})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=400)

        @routes.post("/aerince/lora-recipes/delete")
        async def aerince_lora_recipes_delete(request):
            try:
                data = await request.json()
                recipe_id = str(data.get("id") or "").strip()
                store = _read_lora_recipes()
                before = len(store.get("recipes", []))
                store["recipes"] = [r for r in store.get("recipes", []) if r.get("id") != recipe_id]
                _write_lora_recipes(store)
                return web.json_response({"success": True, "deleted": before - len(store["recipes"])})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=400)

        @routes.get("/aerince/lora-recipes/loras")
        async def aerince_lora_recipes_loras(request):
            names = await asyncio.to_thread(folder_paths.get_filename_list, "loras")
            details = await asyncio.to_thread(
                lambda: [
                    {"name": name, "cover": _lora_cover_asset(name), "created_at": _lora_created_at(name)}
                    for name in names
                ]
            )
            return web.json_response({"success": True, "loras": names, "lora_details": details})

        @routes.get("/aerince/lora-recipes/lora-cover")
        async def aerince_lora_recipes_lora_cover(request):
            name = request.query.get("name", "")
            preview = await asyncio.to_thread(_find_or_cache_lora_preview, name)
            if not preview or not os.path.isfile(preview):
                return web.json_response({"success": False, "error": "cover not found"}, status=404)
            if not _is_lora_path(preview):
                return web.json_response({"success": False, "error": "invalid path"}, status=404)
            return web.FileResponse(preview)

        @routes.post("/aerince/lora-recipes/extract-triggers")
        async def aerince_lora_recipes_extract_triggers(request):
            try:
                data = await request.json()
                trigger_words, details, missing = await asyncio.to_thread(
                    _extract_trigger_words_for_loras,
                    data.get("loras", []),
                    data.get("max_words_per_lora", 1),
                    data.get("separator", ", "),
                    data.get("use_civitai_fallback", True),
                    data.get("civitai_timeout", 10),
                    data.get("filter_common_tags", True),
                    data.get("extra_excluded_tags", ""),
                )
                return web.json_response({"success": True, "trigger_words": trigger_words, "details": details, "missing": missing})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=400)

        @routes.post("/aerince/lora-recipes/cover")
        async def aerince_lora_recipes_cover_upload(request):
            try:
                _ensure_lora_recipe_dirs()
                reader = await request.multipart()
                field = await reader.next()
                if field is None or field.name != "file":
                    return web.json_response({"success": False, "error": "missing file"}, status=400)
                original = os.path.basename(field.filename or "cover.png")
                ext = os.path.splitext(original)[1].lower() or ".png"
                if ext not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
                    return web.json_response({"success": False, "error": f"unsupported cover format: {ext}"}, status=400)
                rel, full = _safe_cover_path(f"covers/{uuid.uuid4().hex}{ext}")
                with open(full, "wb") as out:
                    while True:
                        chunk = await field.read_chunk()
                        if not chunk:
                            break
                        out.write(chunk)
                return web.json_response({"success": True, "cover": rel, "cover_url": _recipe_cover_url(rel)})
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=400)

        @routes.get("/aerince/lora-recipes/cover")
        async def aerince_lora_recipes_cover_get(request):
            try:
                rel = request.query.get("path", "")
                _rel, full = _safe_cover_path(rel)
                if not os.path.exists(full):
                    raise FileNotFoundError(rel)
                return web.FileResponse(full)
            except Exception as e:
                return web.json_response({"success": False, "error": str(e)}, status=404)

        print("[AerinceTools] routes registered")
    except Exception as e:
        print(f"[AerinceTools] route registration skipped: {e}")


# ═══════════════════════════════════════════════
#  HTTP 服务器
# ═══════════════════════════════════════════════
class _BridgeHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        # Silence the frequent /bridge/status polling noise in the ComfyUI console.
        # Uncomment the next line when debugging bridge HTTP requests.
        # print(f"[Bridge HTTP] {args[0]}")
        pass

    def _cors(self, status=200, extra=None):
        self.send_response(status)
        for h, v in (extra or {}).items():
            self.send_header(h, v)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False)
        self._cors(status, {'Content-Type': 'application/json; charset=utf-8',
                            'Content-Length': str(len(body.encode()))})
        self.wfile.write(body.encode())

    def do_OPTIONS(self):
        self._cors(204)

    def do_GET(self):
        if self.path == '/bridge/status':
            self._json(bridge.status())
        else:
            self._json({'success': False, 'error': 'Not found'}, 404)

    def do_POST(self):
        if self.path == '/bridge/workflow':
            try:
                data = json.loads(self._read_body())
            except Exception:
                self._json({'success': False, 'error': 'Invalid JSON'}, 400)
                return

            prompt = data.get('prompt')
            if not isinstance(prompt, dict):
                self._json({'success': False, 'error': 'Missing prompt'}, 400)
                return

            cached_at = bridge.update_prompt(
                prompt,
                data.get('workflow'),
                data.get('client_id'),
            )
            self._json({'success': True, 'cached_at': cached_at})
            return

        if self.path != '/bridge/receive':
            self._json({'success': False, 'error': 'Not found'}, 404)
            return

        try:
            data = json.loads(self._read_body())
        except Exception:
            self._json({'success': False, 'error': 'Invalid JSON'}, 400)
            return

        try:
            image_bytes = None
            source_url = data.get('url', '')
            trigger = data.get('trigger', False)

            raw_b64 = data.get('data', '')
            if raw_b64:
                if ',' in raw_b64 and raw_b64.startswith('data:'):
                    _, raw_b64 = raw_b64.split(',', 1)
                image_bytes = base64.b64decode(raw_b64)
            elif source_url:
                print(f"[Bridge] Downloading: {source_url[:120]}")
                req = Request(source_url, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Referer': f'{urlparse(source_url).scheme}://{urlparse(source_url).netloc}',
                })
                with urlopen(req, timeout=30) as resp:
                    image_bytes = resp.read()
            else:
                self._json({'success': False, 'error': 'No data or url provided'}, 400)
                return

            if not image_bytes:
                self._json({'success': False, 'error': 'Empty image'}, 400)
                return

            img_t, mask_t = _bytes_to_tensor(image_bytes)
            uid = bridge.push(img_t, mask_t, source_url)
            bridge.last_trigger_uid = uid if trigger else bridge.last_trigger_uid

            print(f"[Bridge] Received  {img_t.shape}  uid={uid}")

            trigger_ok = False
            trigger_error = ""
            prompt_id = ""
            if trigger:
                trigger_ok, result = _queue_cached_prompt(uid)
                if trigger_ok:
                    prompt_id = result
                    print(f"[Bridge] Direct queued  uid={uid}  prompt_id={prompt_id}")
                else:
                    trigger_error = result
                    print(f"[Bridge] Direct queue failed: {trigger_error}")
                    ws_ok, ws_error = _send_frontend_trigger(uid)
                    trigger_ok = ws_ok
                    if ws_ok:
                        trigger_error = ""
                        print(f"[Bridge] Frontend trigger fallback sent  uid={uid}")
                    else:
                        trigger_error = f"{trigger_error}; frontend fallback failed: {ws_error}"
                        print(f"[Bridge] Trigger fallback failed: {ws_error}")

            self._json({
                'success': True,
                'uid': uid,
                'shape': list(img_t.shape),
                'trigger_ok': trigger_ok,
                'trigger_error': trigger_error,
                'prompt_id': prompt_id,
            })

        except HTTPError as e:
            self._json({'success': False, 'error': f'Download HTTP {e.code}'}, 400)
        except URLError as e:
            self._json({'success': False, 'error': f'Download: {e.reason}'}, 400)
        except Exception as e:
            traceback.print_exc()
            self._json({'success': False, 'error': str(e)}, 500)

    def _read_body(self):
        n = int(self.headers.get('Content-Length', 0))
        if n > MAX_BODY_SIZE:
            raise ValueError(f'Body too large: {n} bytes')
        return self.rfile.read(n)


def _start_server():
    try:
        srv = HTTPServer(('127.0.0.1', BRIDGE_PORT), _BridgeHandler)
        t = threading.Thread(target=srv.serve_forever, name="bridge-http", daemon=True)
        t.start()
        print(f"[Bridge] HTTP server → http://127.0.0.1:{BRIDGE_PORT}")
        return srv
    except Exception as e:
        print(f"[Bridge] ⚠ HTTP port {BRIDGE_PORT}: {e}")
        return None

_bridge_server = _start_server()


# ═══════════════════════════════════════════════
#  节点 1：BridgeReceiver
#  浏览器直传 + 本地文件 + 透传 + 预览 + 拖放
# ═══════════════════════════════════════════════
class BridgeReceiver:
    """🌉 浏览器直传 / 本地文件 / 透传 — 内置预览"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": (["🌐 浏览器直传", "📁 本地文件"], {"default": "🌐 浏览器直传"}),
                "image": (_get_file_list(), {"image_upload": True}),
                "compress_to_8mb": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "passthrough": ("IMAGE",),
            },
            "hidden": {
                # 隐藏参数用于接收拖放上传的图片
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("image", "mask", "source_url")
    FUNCTION = "receive"
    CATEGORY = "Aerince Tools/Bridge"
    OUTPUT_NODE = False
    DESCRIPTION = "浏览器直传 / 本地文件(可拖放) / IMAGE 透传。超过 8MB 时自动等比缩小。"

    def receive(self, source="🌐 浏览器直传", image="", compress_to_8mb=True, passthrough=None, prompt=None, extra_pnginfo=None):
        _cache_prompt_from_node(prompt, extra_pnginfo)
        img = None
        mask = None
        url = ""

        # 1) 透传
        if passthrough is not None and passthrough.shape[0] > 0:
            img = passthrough
            mask = torch.zeros((passthrough.shape[0], passthrough.shape[1], passthrough.shape[2]),
                               dtype=torch.float32, device=passthrough.device)
            url = "passthrough"
            print(f"[BridgeReceiver] Passthrough {img.shape}")

        # 2) 浏览器直传 — 从队列弹出，保证每张图归属于自己的那次执行
        elif source == "🌐 浏览器直传":
            popped_img, popped_mask, popped_url = bridge.pop()
            if popped_img is not None:
                img = popped_img
                mask = popped_mask
                url = popped_url
                print(f"[BridgeReceiver] Browser (from queue) {img.shape}  url={url[:80]}")

        # 3) 本地文件（下拉或拖放）
        if img is None and image:
            input_dir = folder_paths.get_input_directory()
            fp = os.path.join(input_dir, image)
            if os.path.exists(fp):
                img, mask = _file_to_tensor(fp)
                url = fp
                print(f"[BridgeReceiver] Local file {img.shape} ← {image}")

        # 4) 占位
        if img is None:
            img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            url = ""
            print("[BridgeReceiver] Placeholder")

        if compress_to_8mb:
            old_height, old_width = img.shape[1:3]
            img, mask, encoded_size, compressed = _compress_tensor_to_byte_limit(img, mask)
            if compressed:
                print(
                    f"[BridgeReceiver] Compressed {old_width}x{old_height} -> "
                    f"{img.shape[2]}x{img.shape[1]} ({encoded_size / 1024 / 1024:.2f} MiB PNG)"
                )

        # 预览
        preview = _tensor_to_temp(img, "bridge")

        return {"ui": {"images": [preview]}, "result": (img, mask, url)}

    @classmethod
    def IS_CHANGED(s, source="🌐 浏览器直传", image="", compress_to_8mb=True, passthrough=None, prompt=None, extra_pnginfo=None):
        # 浏览器直传模式：返回队列头部 uid（如果有），确保每张新图触发一次执行
        if source == "🌐 浏览器直传" and passthrough is None:
            return bridge.peek_uid()
        if image:
            fp = os.path.join(folder_paths.get_input_directory(), image)
            if os.path.exists(fp):
                return os.path.getmtime(fp)
        return 0


# ═══════════════════════════════════════════════
#  节点 2：BridgeImageLoader
#  Universal Image Loader 全格式 + 透传输入
# ═══════════════════════════════════════════════
class BridgeImageLoader:
    """📁 Universal Image Loader 全格式 + IMAGE 透传输入"""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (_get_file_list(), {"image_upload": True}),
            },
            "optional": {
                "passthrough": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "load_image"
    CATEGORY = "Aerince Tools/Bridge"
    OUTPUT_NODE = False
    DESCRIPTION = "Universal Image Loader（全格式） + IMAGE 透传。有图连入时优先透传。"

    def load_image(self, image="", passthrough=None, prompt=None, extra_pnginfo=None):
        _cache_prompt_from_node(prompt, extra_pnginfo)
        # 透传优先
        if passthrough is not None and passthrough.shape[0] > 0:
            h, w = passthrough.shape[1], passthrough.shape[2]
            mask = torch.zeros((passthrough.shape[0], h, w),
                               dtype=torch.float32, device=passthrough.device)
            print(f"[BridgeImageLoader] Passthrough {passthrough.shape}")
            # 预览
            preview = _tensor_to_temp(passthrough, "loader")
            return {"ui": {"images": [preview]}, "result": (passthrough, mask)}

        # 加载文件
        if not image:
            img = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            mask = torch.zeros((1, 64, 64), dtype=torch.float32)
            return (img, mask)

        input_dir = folder_paths.get_input_directory()
        fp = os.path.join(input_dir, image)
        if not os.path.exists(fp):
            raise FileNotFoundError(f"File not found: {fp}")

        img, mask = _file_to_tensor(fp)
        print(f"[BridgeImageLoader] Loaded {img.shape} ← {image}")

        # 预览
        preview = _tensor_to_temp(img, "loader")
        return {"ui": {"images": [preview]}, "result": (img, mask)}

    @classmethod
    def IS_CHANGED(s, image="", passthrough=None, prompt=None, extra_pnginfo=None):
        if image:
            fp = os.path.join(folder_paths.get_input_directory(), image)
            if os.path.exists(fp):
                return os.path.getmtime(fp)
        return 0

    @classmethod
    def VALIDATE_INPUTS(s, image="", passthrough=None, prompt=None, extra_pnginfo=None):
        if passthrough is not None:
            return True
        if not image:
            return True
        fp = os.path.join(folder_paths.get_input_directory(), image)
        if not os.path.exists(fp):
            return f"File not found: {image}"
        ext = os.path.splitext(image)[1].lower()
        if ext not in SUPPORTED_FORMATS:
            return f"Unsupported format: {ext}"
        return True


def _read_safetensors_metadata(file_path):
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"LoRA file not found: {file_path}")
    if not file_path.lower().endswith(".safetensors"):
        return {}

    with open(file_path, "rb") as file:
        header_size = int.from_bytes(file.read(8), "little", signed=False)
        if header_size <= 0:
            return {}
        if header_size > min(100 * 1024 * 1024, os.path.getsize(file_path) - 8):
            raise ValueError(f"Invalid safetensors header size: {header_size}")
        header = file.read(header_size)

    header_json = json.loads(header.decode("utf-8"))
    metadata = header_json.get("__metadata__", {})
    if not isinstance(metadata, dict):
        return {}

    normalized = {}
    for key, value in metadata.items():
        normalized[key] = _maybe_json(value)
    return normalized


def _file_sha256(file_path):
    digest = hashlib.sha256()
    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cached_file_sha256(file_path):
    signature = (os.path.abspath(file_path), os.path.getmtime(file_path), os.path.getsize(file_path))
    if signature not in CIVITAI_FILE_HASH_CACHE:
        CIVITAI_FILE_HASH_CACHE[signature] = _file_sha256(file_path)
    return CIVITAI_FILE_HASH_CACHE[signature]


LORA_PREVIEW_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".avif")


def _find_lora_preview(lora_name):
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path:
        return None
    base = os.path.splitext(lora_path)[0]
    for ext in LORA_PREVIEW_EXTS:
        candidate = base + ext
        if os.path.isfile(candidate):
            return candidate
    candidate = base + ".preview.png"
    return candidate if os.path.isfile(candidate) else None


def _lora_cover_url(lora_name):
    return f"/aerince/lora-recipes/lora-cover?name={quote(lora_name)}"


def _is_lora_path(path):
    resolved = os.path.normcase(os.path.abspath(path))
    for folder in folder_paths.get_folder_paths("loras"):
        root = os.path.normcase(os.path.abspath(folder))
        try:
            if os.path.commonpath((resolved, root)) == root:
                return True
        except ValueError:
            pass
    return False


def _read_lora_info_json(lora_path):
    for info_path in (lora_path + ".rgthree-info.json", lora_path + ".weilin-info.json"):
        if os.path.isfile(info_path):
            try:
                with open(info_path, encoding="utf-8-sig", errors="ignore") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    return data
            except Exception:
                pass
    return {}


def _lora_cover_asset(lora_name):
    """Route all covers through the backend so remote images can be cached locally."""
    lora_path = folder_paths.get_full_path("loras", lora_name)
    return _lora_cover_url(lora_name) if lora_path else ""


def _civitai_image_url(data):
    images = data.get("images") if isinstance(data, dict) else None
    if isinstance(images, list):
        for image in images:
            if isinstance(image, dict) and str(image.get("type", "")).lower() != "video" and image.get("url"):
                return str(image["url"])
    return ""


def _download_lora_preview(image_url, target, timeout):
    parsed = urlparse(image_url)
    hostname = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (hostname == "civitai.com" or hostname.endswith(".civitai.com")):
        raise ValueError("unsupported cover URL")
    req = Request(image_url, headers={
        "Accept": "image/*",
        "User-Agent": "Aerince-ComfyUI-LoRA-Cover/1.0",
    })
    with urlopen(req, timeout=int(timeout)) as resp:
        raw = resp.read(25 * 1024 * 1024 + 1)
    if len(raw) > 25 * 1024 * 1024:
        return None
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(raw))).convert("RGB")
    image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    temp = target + f".{uuid.uuid4().hex}.tmp"
    try:
        image.save(temp, format="PNG", optimize=True)
        os.replace(temp, target)
    finally:
        if os.path.exists(temp):
            os.remove(temp)
    return target


def _find_or_cache_lora_preview(lora_name, timeout=15):
    preview = _find_lora_preview(lora_name)
    if preview:
        return preview
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.isfile(lora_path) or not _is_lora_path(lora_path):
        return None
    info_paths = (lora_path + ".rgthree-info.json", lora_path + ".weilin-info.json")
    info_state = tuple(
        (path, os.path.getmtime(path), os.path.getsize(path))
        for path in info_paths if os.path.isfile(path)
    )
    miss_key = (os.path.abspath(lora_path), os.path.getmtime(lora_path), os.path.getsize(lora_path), info_state)
    if miss_key in CIVITAI_COVER_MISS_CACHE:
        return None

    with CIVITAI_COVER_SEMAPHORE:
        preview = _find_lora_preview(lora_name)
        if preview:
            return preview
        target = os.path.splitext(lora_path)[0] + ".preview.png"
        file_hash = _cached_file_sha256(lora_path)
        _, data = _query_civitai_triggers_by_hash(file_hash, timeout)
        image_url = _civitai_image_url(data)
        if not image_url:
            image_url = _civitai_image_url(_read_lora_info_json(lora_path))
        if not image_url:
            if isinstance(data, dict) and not data.get("error"):
                CIVITAI_COVER_MISS_CACHE.add(miss_key)
            return None
        try:
            return _download_lora_preview(image_url, target, timeout)
        except Exception:
            return None


def _lora_created_at(lora_name):
    lora_path = folder_paths.get_full_path("loras", lora_name)
    try:
        return os.path.getctime(lora_path) if lora_path else 0
    except OSError:
        return 0


def _extract_civitai_trigger_words(data):
    if not isinstance(data, dict):
        return []

    candidates = []
    for key in ("trainedWords", "trained_words", "triggerWords", "trigger_words"):
        candidates.extend(_split_trigger_text(data.get(key)))

    for item in data.get("files", []) or []:
        if isinstance(item, dict):
            metadata = item.get("metadata")
            if isinstance(metadata, dict):
                for key in ("trainedWords", "trained_words", "triggerWords", "trigger_words"):
                    candidates.extend(_split_trigger_text(metadata.get(key)))

    return _dedupe_words(candidates)


def _query_civitai_triggers_by_hash(file_hash, timeout=10):
    file_hash = (file_hash or "").strip().lower()
    if not file_hash:
        return [], {}
    if file_hash in CIVITAI_MODEL_VERSION_BY_HASH_CACHE:
        cached = CIVITAI_MODEL_VERSION_BY_HASH_CACHE[file_hash]
        return cached.get("words", []), cached.get("data", {})

    url = f"https://civitai.com/api/v1/model-versions/by-hash/{file_hash}"
    req = Request(url, headers={
        "Accept": "application/json",
        "User-Agent": "Aerince-ComfyUI-LoRA-TriggerWords/1.0",
    }, method="GET")
    try:
        with urlopen(req, timeout=int(timeout)) as resp:
            text = resp.read().decode("utf-8")
            data = json.loads(text) if text else {}
    except Exception as exc:
        data = {"error": str(exc)}

    words = _extract_civitai_trigger_words(data)
    if not isinstance(data, dict) or not data.get("error"):
        CIVITAI_MODEL_VERSION_BY_HASH_CACHE[file_hash] = {
            "words": words,
            "data": data if isinstance(data, dict) else {},
        }
    return words, data if isinstance(data, dict) else {}


def _maybe_json(value):
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    if text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _split_trigger_text(value):
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        words = []
        for item in value:
            words.extend(_split_trigger_text(item))
        return words
    if isinstance(value, dict):
        if "word" in value:
            return _split_trigger_text(value.get("word"))
        words = []
        for item in value.values():
            words.extend(_split_trigger_text(item))
        return words

    text = str(value).strip()
    if not text:
        return []
    text = text.replace("\n", ",").replace("|", ",")
    return [part.strip() for part in text.split(",") if part.strip()]


def _collect_explicit_trigger_words(metadata):
    keys = (
        "trained_words",
        "trainedWords",
        "trigger_words",
        "triggerWords",
        "ss_trigger_words",
        "activation_text",
        "activation text",
        "modelspec.trigger",
        "modelspec.tags",
    )
    words = []
    for key in keys:
        words.extend(_split_trigger_text(metadata.get(key)))
    return _dedupe_words(words)


def _dedupe_words(words):
    seen = set()
    result = []
    for word in words:
        word = str(word).strip()
        if not word:
            continue
        key = word.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(word)
    return result


def _normalize_tag_key(word):
    text = str(word or "").strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"\s+", " ", text)
    return text


def _filter_common_trigger_words(words, extra_excluded_tags=""):
    excluded = set(COMMON_NON_TRIGGER_TAGS)
    for item in _split_trigger_text(extra_excluded_tags):
        key = _normalize_tag_key(item)
        if key:
            excluded.add(key)

    kept = []
    removed = []
    for word in _dedupe_words(words):
        if _normalize_tag_key(word) in excluded:
            removed.append(word)
        else:
            kept.append(word)
    return kept, removed


def _format_words(words, separator):
    separator = separator if separator is not None else ", "
    return separator.join(words)


def _is_rgthree_power_lora_loader(node):
    if not isinstance(node, dict):
        return False
    class_type = str(node.get("class_type", ""))
    title = str((node.get("_meta") or {}).get("title", ""))
    return (
        class_type == "Power Lora Loader (rgthree)"
        or "Power Lora Loader" in class_type
        or "Power Lora Loader" in title
    )


def _normalize_lora_name(lora_name):
    lora_name = (lora_name or "").strip()
    if not lora_name or lora_name.lower() == "none":
        return ""
    if folder_paths.get_full_path("loras", lora_name):
        return lora_name
    if not os.path.splitext(lora_name)[1]:
        candidate = lora_name + ".safetensors"
        if folder_paths.get_full_path("loras", candidate):
            return candidate
    base = os.path.splitext(lora_name.replace("\\", "/"))[0].lower()
    for item in folder_paths.get_filename_list("loras"):
        item_base = os.path.splitext(item.replace("\\", "/"))[0].lower()
        if item.lower() == lora_name.lower() or item_base == base or item_base.endswith("/" + base):
            return item
    return lora_name


def _extract_loras_from_rgthree_prompt_node(node):
    result = []
    inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
    if not isinstance(inputs, dict):
        return result
    for name, value in inputs.items():
        if not str(name).lower().startswith("lora_") or not isinstance(value, dict):
            continue
        if not value.get("on", False):
            continue
        lora_name = _normalize_lora_name(value.get("lora", ""))
        if not lora_name:
            continue
        strength_model = value.get("strength", 1.0)
        strength_clip = value.get("strengthTwo", strength_model)
        result.append({
            "lora": lora_name,
            "strength": strength_model,
            "strength_clip": strength_clip,
        })
    return result


def _get_prompt_nodes(prompt):
    if not isinstance(prompt, dict):
        return {}
    return {str(node_id): node for node_id, node in prompt.items() if isinstance(node, dict)}


def _find_upstream_prompt_node(prompt, unique_id, input_name):
    nodes = _get_prompt_nodes(prompt)
    current = nodes.get(str(unique_id))
    if not current:
        return None
    link = (current.get("inputs") or {}).get(input_name)
    if isinstance(link, (list, tuple)) and link:
        return nodes.get(str(link[0]))
    return None


def _get_linked_prompt_node(nodes, node, input_name):
    if not isinstance(node, dict):
        return None
    link = (node.get("inputs") or {}).get(input_name)
    if isinstance(link, (list, tuple)) and link:
        return nodes.get(str(link[0]))
    return None


def _is_impact_switch(node):
    if not isinstance(node, dict):
        return False
    class_type = str(node.get("class_type", ""))
    title = str((node.get("_meta") or {}).get("title", ""))
    return class_type in {"ImpactSwitch", "LatentSwitch", "SEGSSwitch"} or "Switch" in title


def _resolve_prompt_int(nodes, value, default=1):
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    if isinstance(value, (list, tuple)) and value:
        source = nodes.get(str(value[0]))
        inputs = source.get("inputs", {}) if isinstance(source, dict) else {}
        if isinstance(inputs, dict):
            for key in ("value", "int", "integer", "number"):
                if key in inputs:
                    return _resolve_prompt_int(nodes, inputs[key], default)
    return default


def _selected_impact_switch_input(nodes, node):
    inputs = node.get("inputs", {}) if isinstance(node, dict) else {}
    if not isinstance(inputs, dict):
        return None
    selected_index = _resolve_prompt_int(nodes, inputs.get("select", 1), 1)

    candidates = [
        f"input{selected_index}",
        f"latent{selected_index}",
        f"images{selected_index}_opt",
        f"mask{selected_index}_opt",
    ]
    if selected_index == 1:
        candidates.extend(["images1", "latent1"])
    for input_name in candidates:
        if input_name in inputs:
            return input_name
    return None


def _find_rgthree_loras_from_node(nodes, node, visited=None):
    if visited is None:
        visited = set()
    if not isinstance(node, dict):
        return [], node

    marker = id(node)
    if marker in visited:
        return [], node
    visited.add(marker)

    if _is_rgthree_power_lora_loader(node):
        return _extract_loras_from_rgthree_prompt_node(node), node

    if _is_impact_switch(node):
        selected_input = _selected_impact_switch_input(nodes, node)
        upstream = _get_linked_prompt_node(nodes, node, selected_input) if selected_input else None
        if upstream:
            return _find_rgthree_loras_from_node(nodes, upstream, visited)

    return [], node


def _find_rgthree_loras(prompt, unique_id):
    upstream = _find_upstream_prompt_node(prompt, unique_id, "source")
    nodes = _get_prompt_nodes(prompt)
    if upstream:
        loras, source_node = _find_rgthree_loras_from_node(nodes, upstream)
        if loras or _is_rgthree_power_lora_loader(source_node) or _is_impact_switch(upstream):
            return loras, source_node

    for node in nodes.values():
        if _is_rgthree_power_lora_loader(node):
            loras = _extract_loras_from_rgthree_prompt_node(node)
            if loras:
                return loras, node
    return [], upstream


def _trigger_words_for_lora(
    lora_name,
    max_words,
    use_civitai_fallback=False,
    civitai_timeout=10,
    filter_common_tags=True,
    extra_excluded_tags="",
):
    lora_path = folder_paths.get_full_path("loras", lora_name)
    if not lora_path or not os.path.exists(lora_path):
        return [], {}, {
            "source": "missing_file",
            "file_hash": "",
            "civitai": {},
            "filtered_words": [],
        }

    metadata = _read_safetensors_metadata(lora_path)
    words = _collect_explicit_trigger_words(metadata)
    source = "metadata" if words else ""
    if not words:
        words = _collect_explicit_trigger_words(_read_lora_info_json(lora_path))
        source = "info_json" if words else ""

    file_hash = ""
    civitai_info = {}
    if not words and use_civitai_fallback:
        try:
            file_hash = _cached_file_sha256(lora_path)
            civitai_words, civitai_data = _query_civitai_triggers_by_hash(file_hash, int(civitai_timeout))
            words = civitai_words
            source = "civitai" if words else "civitai_empty"
            civitai_model = civitai_data.get("model") if isinstance(civitai_data, dict) else {}
            if not isinstance(civitai_model, dict):
                civitai_model = {}
            civitai_info = {
                "model_id": civitai_data.get("modelId") if isinstance(civitai_data, dict) else None,
                "model_version_id": civitai_data.get("id") if isinstance(civitai_data, dict) else None,
                "model_name": civitai_model.get("name"),
                "version_name": civitai_data.get("name") if isinstance(civitai_data, dict) else None,
                "error": civitai_data.get("error") if isinstance(civitai_data, dict) else None,
            }
        except Exception as exc:
            source = "civitai_error"
            civitai_info = {"error": str(exc)}

    words = _dedupe_words(words)
    filtered_words = []
    if filter_common_tags:
        words, filtered_words = _filter_common_trigger_words(words, extra_excluded_tags)

    return words[:int(max_words)], metadata, {
        "source": source or "none",
        "file_hash": file_hash,
        "civitai": civitai_info,
        "filtered_words": filtered_words,
    }


class AerinceLoRARecipeManager:
    """Enhanced multi-LoRA stack loader with recipe persistence handled by the frontend/API."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "lora_stack": ("STRING", {
                    "default": "[]",
                    "multiline": False,
                }),
                "trigger_words": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "recipe_id": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "recipe_name": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    RETURN_NAMES = ("model", "clip", "trigger_words")
    FUNCTION = "apply_lora_stack"
    CATEGORY = "Aerince Tools/LoRA"
    OUTPUT_NODE = False
    DESCRIPTION = "Enhanced LoRA recipe manager: stack multiple LoRAs with one synced strength, save recipes from the frontend, and output LoRA trigger words."

    def _parse_lora_stack(self, lora_stack):
        if isinstance(lora_stack, str):
            text = lora_stack.strip()
            if not text:
                return []
            try:
                data = json.loads(text)
            except Exception as exc:
                print(f"[AerinceTools] LoRA recipe manager invalid lora_stack JSON: {exc}")
                return []
        else:
            data = lora_stack
        return _normalize_recipe_loras(data if isinstance(data, list) else [])

    def apply_lora_stack(
        self,
        model,
        clip,
        lora_stack="[]",
        trigger_words="",
        recipe_id="",
        recipe_name="",
    ):
        del recipe_id, recipe_name
        loras = self._parse_lora_stack(lora_stack)
        applied = []
        missing = []

        for lora in loras:
            if not lora.get("enabled", True):
                continue
            name = lora.get("name", "")
            path = folder_paths.get_full_path("loras", name)
            if not path or not os.path.exists(path):
                missing.append(name)
                print(f"[AerinceTools] LoRA recipe manager warning: missing LoRA skipped: {name}")
                continue
            try:
                strength = float(lora.get("strength", 1.0))
                lora_sd = comfy.utils.load_torch_file(path, safe_load=True)
                model, clip = comfy.sd.load_lora_for_models(model, clip, lora_sd, strength, strength)
                applied.append(name)
            except Exception as exc:
                missing.append(name)
                print(f"[AerinceTools] LoRA recipe manager warning: failed to load LoRA skipped: {name}: {exc}")

        output_triggers = str(trigger_words or "").strip()
        ui_text = [
            f"Applied LoRAs: {', '.join(applied) if applied else '(none)'}",
            f"Trigger words: {output_triggers}",
        ]
        if missing:
            ui_text.append(f"Missing/skipped LoRAs: {', '.join(_dedupe_words(missing))}")
        return {"ui": {"text": ui_text}, "result": (model, clip, output_triggers)}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        state = {
            "lora_stack": kwargs.get("lora_stack", ""),
            "trigger_words": kwargs.get("trigger_words", ""),
        }
        try:
            loras = json.loads(kwargs.get("lora_stack") or "[]")
        except Exception:
            loras = []
        mtimes = []
        for lora in _normalize_recipe_loras(loras):
            path = folder_paths.get_full_path("loras", lora.get("name", ""))
            mtimes.append((lora.get("name"), os.path.getmtime(path) if path and os.path.exists(path) else 0))
        state["mtimes"] = mtimes
        return json.dumps(state, ensure_ascii=False)


class LoRAStackTriggerWords:
    """Read enabled LoRAs from an upstream rgthree Power Lora Loader and output trigger words."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "max_words_per_lora": ("INT", {
                    "default": 20,
                    "min": 1,
                    "max": 200,
                    "step": 1,
                }),
                "separator": ("STRING", {
                    "default": ", ",
                    "multiline": False,
                }),
                "include_lora_tags": ("BOOLEAN", {
                    "default": False,
                }),
                "use_civitai_fallback": ("BOOLEAN", {
                    "default": True,
                }),
                "civitai_timeout": ("INT", {
                    "default": 10,
                    "min": 1,
                    "max": 60,
                    "step": 1,
                }),
                "filter_common_tags": ("BOOLEAN", {
                    "default": True,
                }),
                "extra_excluded_tags": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
            },
            "optional": {
                "source": (any_type,),
            },
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("trigger_words", "prompt", "lora_names", "details_json")
    FUNCTION = "read_trigger_words"
    CATEGORY = "Aerince Tools/LoRA"
    OUTPUT_NODE = False
    DESCRIPTION = "Connect source to rgthree Power Lora Loader output. This node reads enabled LoRAs from the workflow prompt and emits metadata trigger words."

    def read_trigger_words(
        self,
        max_words_per_lora=20,
        separator=", ",
        include_lora_tags=False,
        use_civitai_fallback=True,
        civitai_timeout=10,
        filter_common_tags=True,
        extra_excluded_tags="",
        source=None,
        prompt=None,
        unique_id=None,
    ):
        del source
        loras, source_node = _find_rgthree_loras(prompt, unique_id)
        all_words = []
        prompt_parts = []
        details = []

        for lora in loras:
            lora_name = lora["lora"]
            words, metadata, lookup = _trigger_words_for_lora(
                lora_name,
                int(max_words_per_lora),
                bool(use_civitai_fallback),
                int(civitai_timeout),
                bool(filter_common_tags),
                extra_excluded_tags,
            )
            all_words.extend(words)

            if include_lora_tags:
                lora_base = os.path.splitext(lora_name.replace("\\", "/"))[0]
                strength_model = float(lora.get("strength", 1.0))
                strength_clip = float(lora.get("strength_clip", strength_model))
                prompt_parts.append(f"<lora:{lora_base}:{strength_model:g}:{strength_clip:g}>")
            if words:
                prompt_parts.append(_format_words(words, separator))

            details.append({
                "lora": lora_name,
                "strength": lora.get("strength", 1.0),
                "strength_clip": lora.get("strength_clip", lora.get("strength", 1.0)),
                "trigger_words": words,
                "trigger_source": lookup.get("source"),
                "sha256": lookup.get("file_hash"),
                "civitai": lookup.get("civitai"),
                "filtered_words": lookup.get("filtered_words"),
                "metadata_keys": sorted(metadata.keys()) if isinstance(metadata, dict) else [],
            })

        trigger_words = _format_words(_dedupe_words(all_words), separator)
        prompt_text = _format_words([part for part in prompt_parts if part], separator)
        lora_names = _format_words([item["lora"] for item in loras], separator)
        details_json = json.dumps({
            "source_node_id": source_node.get("id") if isinstance(source_node, dict) else None,
            "source_node_title": (source_node.get("_meta") or {}).get("title") if isinstance(source_node, dict) else None,
            "loras": details,
        }, ensure_ascii=False, indent=2)

        return {"ui": {"text": [trigger_words]}, "result": (trigger_words, prompt_text, lora_names, details_json)}

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        prompt = kwargs.get("prompt")
        unique_id = kwargs.get("unique_id")
        loras, _source_node = _find_rgthree_loras(prompt, unique_id)
        state = []
        state.append((
            "settings",
            kwargs.get("max_words_per_lora"),
            kwargs.get("include_lora_tags"),
            kwargs.get("use_civitai_fallback"),
            kwargs.get("civitai_timeout"),
            kwargs.get("filter_common_tags"),
            kwargs.get("extra_excluded_tags"),
        ))
        for lora in loras:
            lora_path = folder_paths.get_full_path("loras", lora["lora"])
            mtime = os.path.getmtime(lora_path) if lora_path and os.path.exists(lora_path) else 0
            state.append((lora.get("lora"), lora.get("strength"), lora.get("strength_clip"), mtime))
        return json.dumps(state, ensure_ascii=False)


class LLMImagePromptReverse:
    """Call an OpenAI-compatible vision chat model to describe an image."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "base_url": ("STRING", {
                    "default": "https://ark.cn-beijing.volces.com/api/v3",
                    "multiline": False,
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "model": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "system_prompt": ("STRING", {
                    "default": "You are an expert image prompt engineer. Analyze the image and return a detailed image generation prompt. Describe subject, composition, style, lighting, colors, background, camera, quality tags, and notable details. Return only the final prompt text. Do not include thinking, reasoning, analysis, explanations, labels, markdown, or XML tags.",
                    "multiline": True,
                }),
                "user_prompt": ("STRING", {
                    "default": "Reverse this image into a high-quality prompt.",
                    "multiline": True,
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "timeout": ("INT", {
                    "default": 300,
                    "min": 5,
                    "max": 900,
                    "step": 1,
                }),
                "thinking_strength": (list(THINKING_STRENGTH_OPTIONS), {
                    "default": "auto",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "reverse_prompt"
    CATEGORY = "Aerince Tools/LLM"
    OUTPUT_NODE = False
    DESCRIPTION = "Use an OpenAI-compatible LLM endpoint to reverse an input image into prompt text. Model capability is left to the user."

    def reverse_prompt(
        self,
        image,
        base_url,
        api_key,
        model,
        system_prompt,
        user_prompt,
        temperature=0.2,
        timeout=300,
        thinking_strength="auto",
        **legacy_options,
    ):
        text = _llm_reverse_image_prompt_from_tensor(
            image,
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            timeout=timeout,
            thinking_strength=thinking_strength,
        )
        return (text,)


class LLMPromptOptimizer:
    """Call an OpenAI-compatible chat model to optimize prompt text."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "prompt": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "base_url": ("STRING", {
                    "default": "https://ark.cn-beijing.volces.com/api/v3",
                    "multiline": False,
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "model": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "system_prompt": ("STRING", {
                    "default": "You are an expert image prompt optimizer. Rewrite the user's prompt into a cleaner, richer, production-ready image generation prompt. Preserve the original intent, improve visual detail, composition, lighting, style, camera language, and quality tags. Return only the final optimized prompt text. Do not include thinking, reasoning, analysis, explanations, labels, markdown, or XML tags.",
                    "multiline": True,
                }),
                "user_prompt": ("STRING", {
                    "default": "Optimize this image generation prompt:",
                    "multiline": True,
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.4,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "timeout": ("INT", {
                    "default": 300,
                    "min": 5,
                    "max": 900,
                    "step": 1,
                }),
                "thinking_strength": (list(THINKING_STRENGTH_OPTIONS), {
                    "default": "auto",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("optimized_prompt",)
    FUNCTION = "optimize_prompt"
    CATEGORY = "Aerince Tools/LLM"
    OUTPUT_NODE = False
    DESCRIPTION = "Use an OpenAI-compatible LLM endpoint to optimize text prompts. Model choice is left to the user."

    def optimize_prompt(
        self,
        prompt,
        base_url,
        api_key,
        model,
        system_prompt,
        user_prompt,
        temperature=0.4,
        timeout=300,
        thinking_strength="auto",
        **legacy_options,
    ):
        base_url, api_key = resolve_llm_provider(base_url, api_key)
        model = (model or "").strip()
        if not model or model == "refresh_models_first":
            raise ValueError("model is required. Click refresh_models and choose a model.")

        prompt = prompt or ""
        url = _normalize_openai_base_url(base_url) + "/chat/completions"
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": f"{user_prompt or 'Optimize this prompt:'}\n\n{prompt}"},
            ],
            "temperature": float(temperature),
        }
        _apply_thinking_options(payload, model, thinking_strength)

        data = _request_chat_json(
            url,
            api_key=api_key,
            payload=payload,
            timeout=int(timeout),
        )
        return (_extract_chat_text(data),)



class LLMImagePromptReverseEditable:
    """Manual reverse button + editable prompt box; smart auto reverse on image change."""

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "base_url": ("STRING", {
                    "default": "https://ark.cn-beijing.volces.com/api/v3",
                    "multiline": False,
                }),
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "model": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
                "system_prompt": ("STRING", {
                    "default": "You are an expert image prompt engineer. Analyze the image and return a detailed image generation prompt. Describe subject, composition, style, lighting, colors, background, camera, quality tags, and notable details. Return only the final prompt text. Do not include thinking, reasoning, analysis, explanations, labels, markdown, or XML tags.",
                    "multiline": True,
                }),
                "user_prompt": ("STRING", {
                    "default": "Reverse this image into a high-quality prompt.",
                    "multiline": True,
                }),
                "prompt_text": ("STRING", {
                    "default": "",
                    "multiline": True,
                }),
                "auto_reverse": ("BOOLEAN", {
                    "default": False,
                    "label_on": "auto when image changes",
                    "label_off": "manual only",
                }),
                # Hidden cache: hash of the image that produced prompt_text.
                "source_image_hash": ("STRING", {
                    "default": "",
                    "multiline": False,
                }),
            },
            "optional": {
                "temperature": ("FLOAT", {
                    "default": 0.2,
                    "min": 0.0,
                    "max": 2.0,
                    "step": 0.05,
                }),
                "timeout": ("INT", {
                    "default": 300,
                    "min": 5,
                    "max": 900,
                    "step": 1,
                }),
                "thinking_strength": (list(THINKING_STRENGTH_OPTIONS), {
                    "default": "auto",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("prompt",)
    FUNCTION = "output_prompt"
    CATEGORY = "Aerince Tools/LLM"
    OUTPUT_NODE = False
    DESCRIPTION = "Reverse via button into an editable box. With auto_reverse on, reverse only when the input image changes; otherwise reuse prompt_text."

    def output_prompt(
        self,
        image,
        base_url,
        api_key,
        model,
        system_prompt,
        user_prompt,
        prompt_text,
        auto_reverse=False,
        source_image_hash="",
        temperature=0.2,
        timeout=300,
        thinking_strength="auto",
        **legacy_options,
    ):
        current_text = prompt_text or ""
        cached_hash = (source_image_hash or "").strip()

        if not bool(auto_reverse):
            return (current_text,)

        current_hash = _image_tensor_fingerprint(image)
        # Cache hit: same image as last reverse/edit source, keep existing text.
        if current_hash and current_hash == cached_hash and current_text.strip():
            return (current_text,)

        text = _llm_reverse_image_prompt_from_tensor(
            image,
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            timeout=timeout,
            thinking_strength=thinking_strength,
        )
        return {
            "ui": {
                "prompt_text": [text],
                "source_image_hash": [current_hash],
            },
            "result": (text,),
        }


class AerinceAutoFreeModelAndNodeCache:
    """Request ComfyUI's native post-prompt model unload and node-cache reset."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "trigger": (any_type,),
            }
        }

    RETURN_TYPES = (any_type,)
    RETURN_NAMES = ("passthrough",)
    FUNCTION = "free_after_prompt"
    CATEGORY = "Aerince Tools/Memory"
    OUTPUT_NODE = True
    DESCRIPTION = (
        "在本次工作流全部执行完成后，请求 ComfyUI 原生卸载模型并清空节点缓存。"
        "把最终 IMAGE 或其他末端输出接到 trigger，确保节点属于执行链。"
    )

    def free_after_prompt(self, enabled, trigger):
        if enabled:
            from server import PromptServer

            prompt_server = PromptServer.instance
            if prompt_server is None or prompt_server.prompt_queue is None:
                raise RuntimeError("ComfyUI PromptServer/prompt queue is not ready")

            # This mirrors POST /free with both options enabled. main.py consumes
            # the flags only after PromptExecutor.execute() has returned, so cache
            # reset cannot invalidate tensors while this prompt is still running.
            prompt_server.prompt_queue.set_flag("unload_models", True)
            prompt_server.prompt_queue.set_flag("free_memory", True)
            print("[Aerince Auto Free] Scheduled model unload and node-cache reset after this prompt.")

        return (trigger,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # NaN is the standard ComfyUI marker for an output node that must run on
        # every queued prompt instead of reusing its cached result.
        return float("NaN")


# ═══════════════════════════════════════════════
#  注册
# ═══════════════════════════════════════════════
_register_aerince_routes()

NODE_CLASS_MAPPINGS = {
    "BridgeReceiver": BridgeReceiver,
    "BridgeImageLoader": BridgeImageLoader,
    "AerinceLoRARecipeManager": AerinceLoRARecipeManager,
    "LoRAStackTriggerWords": LoRAStackTriggerWords,
    "LLMImagePromptReverse": LLMImagePromptReverse,
    "LLMImagePromptReverseEditable": LLMImagePromptReverseEditable,
    "LLMPromptOptimizer": LLMPromptOptimizer,
    "AerinceAutoFreeModelAndNodeCache": AerinceAutoFreeModelAndNodeCache,
    "AerinceCLTaggerV2": AerinceCLTaggerV2,
    "AerinceDLSS5Image": AerinceDLSS5Image,
    "AerinceLocalLLMImagePromptReverse": AerinceLocalLLMImagePromptReverse,
    "AerinceLocalLLMTextProcess": AerinceLocalLLMTextProcess,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BridgeReceiver": "🌉 Bridge Receiver (直传+本地+预览)",
    "BridgeImageLoader": "📁 Bridge Image Loader (全格式+透传)",
    "AerinceLoRARecipeManager": "Aerince LoRA 配方管理器",
    "LoRAStackTriggerWords": "Aerince LoRA Stack Trigger Words",
    "LLMImagePromptReverse": "Aerince LLM Image Prompt Reverse",
    "LLMImagePromptReverseEditable": "Aerince LLM 图像反推（可编辑）",
    "LLMPromptOptimizer": "Aerince LLM Prompt Optimizer",
    "AerinceAutoFreeModelAndNodeCache": "Aerince 自动释放模型与节点缓存",
    "AerinceCLTaggerV2": "Aerince Tagger",
    "AerinceDLSS5Image": "Aerince DLSS 5 Image Enhancer",
    "AerinceLocalLLMImagePromptReverse": "Aerince 本地大模型图像反推",
    "AerinceLocalLLMTextProcess": "Aerince 本地大模型文本处理",
}

print(f"[BridgePack] loaded OK — {list(NODE_CLASS_MAPPINGS.keys())}")
