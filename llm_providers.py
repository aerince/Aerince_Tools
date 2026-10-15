"""Local LLM providers. Public records never contain credentials."""
import json
import os
import tempfile
import threading
import uuid
from pathlib import Path
from urllib.parse import urlsplit


PROVIDER_FILE = Path(__file__).resolve().parent / "user_data" / "llm_providers.json"
PROVIDER_PREFIX = "aerince-provider://"
_LOCK = threading.RLock()


def _read_providers():
    if not PROVIDER_FILE.exists():
        return []
    try:
        data = json.loads(PROVIDER_FILE.read_text(encoding="utf-8"))
        providers = data["providers"]
        if not isinstance(providers, list) or any(
            not isinstance(item, dict) or any(not isinstance(item.get(key), str)
                                            for key in ("id", "name", "base_url", "api_key"))
            for item in providers
        ):
            raise ValueError("Invalid provider data")
        return providers
    except (ValueError, KeyError, TypeError):
        raise ValueError("提供商配置文件损坏，已停止操作以避免覆盖原数据。") from None


def _public_provider(provider):
    return {key: provider[key] for key in ("id", "name", "base_url")} | {"has_key": bool(provider["api_key"])}


def list_llm_providers():
    with _LOCK:
        return [_public_provider(item) for item in _read_providers()]


def save_llm_provider(data):
    if not isinstance(data, dict):
        raise ValueError("提供商配置必须是 JSON 对象。")
    name, base_url = data.get("name"), data.get("base_url")
    provider_id = data.get("id", "")
    if not isinstance(name, str) or not name.strip() or len(name.strip()) > 100:
        raise ValueError("请输入提供商名称（1–100 字符）。")
    if not isinstance(provider_id, str):
        raise ValueError("提供商 ID 无效。")
    if not isinstance(base_url, str):
        raise ValueError("请输入 API 地址。")
    base_url = base_url.strip().rstrip("/")
    try:
        parsed = urlsplit(base_url)
        _ = parsed.port  # Validate malformed/out-of-range ports before saving.
        valid = (parsed.scheme in ("http", "https") and parsed.hostname and
                 not parsed.username and not parsed.password and not parsed.query and not parsed.fragment and
                 not any(character.isspace() or ord(character) < 32 for character in base_url))
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("API 地址需要有效的 http(s) URL，不能包含账号密码、查询参数或片段。")
    if "api_key" in data and not isinstance(data["api_key"], str):
        raise ValueError("API Key 必须是字符串；留空可用于无需认证的本地接口。")
    with _LOCK:
        providers = _read_providers()
        existing = next((item for item in providers if item["id"] == provider_id), None)
        if provider_id and existing is None:
            raise ValueError("提供商不存在，请刷新列表后重试。")
        if any(item["name"].casefold() == name.strip().casefold() and item["id"] != provider_id for item in providers):
            raise ValueError("已有同名提供商，请选择它进行编辑或更换名称。")
        provider = {"id": provider_id or uuid.uuid4().hex, "name": name.strip(), "base_url": base_url,
                    "api_key": data.get("api_key", existing["api_key"] if existing else "").strip()}
        if "\r" in provider["api_key"] or "\n" in provider["api_key"]:
            raise ValueError("API Key 不能包含换行符。")
        updated = [provider if item["id"] == provider_id else item for item in providers]
        if existing is None:
            updated.append(provider)
        PROVIDER_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            # Atomic replace: a failed save leaves the previous configuration intact.
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=PROVIDER_FILE.parent,
                                             prefix=".llm-providers-", suffix=".tmp", delete=False) as file:
                temporary = file.name
                json.dump({"version": 1, "providers": updated}, file, ensure_ascii=False, indent=2)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, PROVIDER_FILE)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
        return _public_provider(provider)


def resolve_llm_provider(base_url, api_key=""):
    value = (base_url or "").strip()
    if not value.startswith(PROVIDER_PREFIX):
        return base_url, api_key
    provider_id = value[len(PROVIDER_PREFIX):]
    with _LOCK:
        provider = next((item for item in _read_providers() if item["id"] == provider_id), None)
        if provider is None:
            raise ValueError("当前工作流引用的提供商不存在，请打开提供商窗口重新选择。")
        return provider["base_url"], provider["api_key"]
