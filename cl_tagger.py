"""Aerince Tagger: CL ONNX and WD EVA02 safetensors image-to-tags node."""

import csv
import json
from pathlib import Path

import folder_paths
import numpy as np
from PIL import Image


MODEL_DIR = Path(folder_paths.models_dir) / "cl_tagger_v2"
MODEL_URL = "https://huggingface.co/cella110n/cl_tagger_v2"
WD_MODEL_REPO = "ashen-sensored/wd-eva02-tagger-2026-canary"
WD_MODEL_NAME = "wd-eva02-tagger-2026-canary/model.safetensors"
WD_MODEL_DIR = Path(folder_paths.models_dir) / "wd_tagger" / "wd-eva02-tagger-2026-canary"
WD_MODEL_FILES = ("config.json", "selected_tags.csv", "model.safetensors")
_SESSION_CACHE = {}
_TORCH_MODEL_CACHE = {}


def _models():
    cl_models = {
        path.relative_to(MODEL_DIR).as_posix(): path
        for path in sorted(MODEL_DIR.glob("**/model.onnx"))
        if (path.parent / "model.onnx.data").is_file()
        and (path.parent / "model_vocabulary.json").is_file()
    }
    return {WD_MODEL_NAME: WD_MODEL_DIR / "model.safetensors", **cl_models}


def _preprocess(image):
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    resized = Image.fromarray(pixels, "RGB").resize((384, 384), Image.Resampling.BICUBIC)
    normalized = (np.asarray(resized, dtype=np.float32) / 255.0 - 0.5) / 0.5
    return np.ascontiguousarray(normalized.transpose(2, 0, 1)[None])


def _preprocess_wd(image, transform):
    pixels = np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)
    pil_image = Image.fromarray(pixels)
    if pil_image.mode == "RGBA":
        background = Image.new("RGBA", pil_image.size, (255, 255, 255, 255))
        background.alpha_composite(pil_image)
        pil_image = background.convert("RGB")
    else:
        pil_image = pil_image.convert("RGB")
    width, height = pil_image.size
    size = max(width, height)
    canvas = Image.new("RGB", (size, size), (255, 255, 255))
    canvas.paste(pil_image, ((size - width) // 2, (size - height) // 2))
    # WD models are trained on BGR, unlike the existing CL RGB pipeline.
    return transform(canvas).unsqueeze(0)[:, [2, 1, 0]]


def _select_tags(logits, vocabulary, threshold, replace_space, exclude_tags, character_threshold=None):
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
    names = vocabulary["idx_to_tag"]
    excluded = tuple(tag.strip().casefold().replace("_", " ") for tag in exclude_tags.split(",") if tag.strip())
    selected = []
    categories = vocabulary.get("categories")
    character_threshold = threshold if character_threshold is None else character_threshold
    for index in np.flatnonzero(probabilities >= min(threshold, character_threshold)):
        cutoff = threshold
        if categories is not None:
            category = categories[index]
            if category not in (0, 4):
                continue
            if category == 4:
                cutoff = character_threshold
        if probabilities[index] < cutoff:
            continue
        tag = names[str(int(index))]
        if any(item in tag.casefold().replace("_", " ") for item in excluded):
            continue
        selected.append((tag, float(probabilities[index])))
    selected.sort(key=lambda item: item[1], reverse=True)
    return ", ".join(tag.replace("_", " ") if replace_space else tag for tag, _ in selected)


def _wd_vocabulary(path, num_classes):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != num_classes:
        raise RuntimeError("WD Tagger model and vocabulary versions do not match.")
    return {
        "idx_to_tag": {str(index): row["name"] for index, row in enumerate(rows)},
        "num_tags": len(rows),
        "categories": [int(row["category"]) for row in rows],
    }


def _wd_model(model_path):
    key = str(model_path)
    if key not in _TORCH_MODEL_CACHE:
        import timm
        from safetensors.torch import load_file
        from timm.data import create_transform, resolve_data_config

        for filename in WD_MODEL_FILES:
            if not (model_path.parent / filename).is_file():
                from huggingface_hub import hf_hub_download

                print(f"[Aerince Tagger] Downloading {filename} from {WD_MODEL_REPO} ...")
                hf_hub_download(repo_id=WD_MODEL_REPO, filename=filename, local_dir=str(model_path.parent))
        with (model_path.parent / "config.json").open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        vocabulary = _wd_vocabulary(model_path.parent / "selected_tags.csv", config["num_classes"])
        _TORCH_MODEL_CACHE.clear()
        _SESSION_CACHE.clear()
        model = timm.create_model(
            config["architecture"],
            pretrained=False,
            num_classes=config["num_classes"],
            global_pool=config.get("global_pool", "avg"),
            **config.get("model_args", {}),
        )
        model.load_state_dict(load_file(str(model_path), device="cpu"), strict=True)
        model.eval()
        transform = create_transform(**resolve_data_config(config["pretrained_cfg"], model=model))
        _TORCH_MODEL_CACHE[key] = (model, transform, vocabulary)
    return _TORCH_MODEL_CACHE[key]


def _tag_wd(image, model_path, threshold, provider, replace_space, exclude_tags, character_threshold):
    import torch

    if provider == "CUDA" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; use Auto or CPU.")
    device = torch.device("cuda" if provider != "CPU" and torch.cuda.is_available() else "cpu")
    model, transform, vocabulary = _wd_model(model_path)
    if device.type == "cuda":
        import comfy.model_management as model_management

        model_management.free_memory(
            model_management.module_size(model) + model_management.minimum_inference_memory(), device
        )
    results = []
    try:
        model.to(device)
        with torch.inference_mode():
            for item in image:
                inputs = _preprocess_wd(item.detach().cpu().numpy(), transform).to(device)
                logits = model(inputs)[0].float().cpu().numpy()
                if len(logits) != vocabulary["num_tags"]:
                    raise RuntimeError("WD Tagger model and vocabulary versions do not match.")
                results.append(
                    _select_tags(logits, vocabulary, threshold, replace_space, exclude_tags, character_threshold)
                )
    finally:
        if device.type != "cpu":
            model.cpu()
    return "\n".join(results), results


def _session(model_path, provider):
    import onnxruntime as ort

    available = ort.get_available_providers()
    if provider == "CUDA" and "CUDAExecutionProvider" not in available:
        raise RuntimeError("CUDAExecutionProvider is unavailable; use Auto or CPU.")
    providers = (
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if provider == "CUDA" or (provider == "Auto" and "CUDAExecutionProvider" in available)
        else ["CPUExecutionProvider"]
    )
    key = (str(model_path), tuple(providers))
    if key not in _SESSION_CACHE:
        try:
            _TORCH_MODEL_CACHE.clear()
            _SESSION_CACHE.clear()
            _SESSION_CACHE[key] = ort.InferenceSession(str(model_path), providers=providers)
        except Exception:
            if provider != "Auto" or providers == ["CPUExecutionProvider"]:
                raise
            providers = ["CPUExecutionProvider"]
            key = (str(model_path), tuple(providers))
            _SESSION_CACHE[key] = ort.InferenceSession(str(model_path), providers=providers)
    return _SESSION_CACHE[key]


class AerinceCLTaggerV2:
    @classmethod
    def INPUT_TYPES(cls):
        models = list(_models())
        return {
            "required": {
                "image": ("IMAGE",),
                "model_name": (models or ["model files not found"],),
                "threshold": ("FLOAT", {"default": 0.6094, "min": 0.05, "max": 0.95, "step": 0.01}),
                "provider": (["Auto", "CUDA", "CPU"], {"default": "Auto"}),
                "replace_space": ("BOOLEAN", {"default": True}),
                "exclude_tags": ("STRING", {"default": "", "multiline": False}),
            },
            "optional": {
                "character_threshold": ("FLOAT", {"default": 0.75, "min": 0.05, "max": 0.95, "step": 0.01}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("tags", "tags_list")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "tag"
    CATEGORY = "Aerince Tools/Tagger"
    DESCRIPTION = "Danbooru tags from CL Tagger v2 or WD EVA02 2026 Canary. Missing WD model files download on first use."

    def tag(self, image, model_name, threshold, provider, replace_space, exclude_tags, character_threshold=0.75):
        models = _models()
        if model_name not in models:
            raise FileNotFoundError(
                f"CL Tagger v2 model not found. Download model.onnx, model.onnx.data and "
                f"model_vocabulary.json from {MODEL_URL} into {MODEL_DIR / 'v2_01a'}."
            )
        model_path = models[model_name]
        if model_name == WD_MODEL_NAME:
            return _tag_wd(image, model_path, threshold, provider, replace_space, exclude_tags, character_threshold)
        with (model_path.parent / "model_vocabulary.json").open("r", encoding="utf-8") as handle:
            vocabulary = json.load(handle)
        session = _session(model_path, provider)
        results = []
        for item in image:
            logits = session.run(["logits"], {"pixel_values": _preprocess(item.detach().cpu().numpy())})[0][0]
            if len(logits) != int(vocabulary["num_tags"]):
                raise RuntimeError("CL Tagger model and vocabulary versions do not match.")
            results.append(_select_tags(logits, vocabulary, threshold, replace_space, exclude_tags))
        return "\n".join(results), results


NODE_CLASS_MAPPINGS = {"AerinceCLTaggerV2": AerinceCLTaggerV2}
NODE_DISPLAY_NAME_MAPPINGS = {"AerinceCLTaggerV2": "Aerince Tagger"}
