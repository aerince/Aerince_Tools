"""ComfyUI node for the local DLSS 5 Visual Enhancer runtime."""

from __future__ import annotations

import importlib.util
import sys
import threading
import types
from pathlib import Path

import cv2
import numpy as np
import torch


DLSS_ROOT = Path(r"D:\Program Files\AI\DLSS.5.Visual.Enhancer.v3.0")
_DLSS_PACKAGE = "_aerince_dlss5_runtime"
_DLSS_LOCK = threading.Lock()


def _load_runtime_modules():
    """Load the bundled src package without shadowing other Python packages named src."""
    package_dir = DLSS_ROOT / "src"
    init_file = package_dir / "__init__.py"
    if not init_file.is_file():
        raise FileNotFoundError(f"DLSS 5 runtime package not found: {package_dir}")

    package = sys.modules.get(_DLSS_PACKAGE)
    if package is None:
        # The bundled __init__.py imports optional image codecs. The node only
        # needs runtime.py and video.py, so create a namespace package instead.
        package = types.ModuleType(_DLSS_PACKAGE)
        package.__file__ = str(init_file)
        package.__path__ = [str(package_dir)]
        package.__package__ = _DLSS_PACKAGE
        sys.modules[_DLSS_PACKAGE] = package

    runtime_name = f"{_DLSS_PACKAGE}.runtime"
    video_name = f"{_DLSS_PACKAGE}.video"
    runtime = sys.modules.get(runtime_name)
    video = sys.modules.get(video_name)
    if runtime is None:
        runtime_spec = importlib.util.spec_from_file_location(
            runtime_name, package_dir / "runtime.py"
        )
        if runtime_spec is None or runtime_spec.loader is None:
            raise ImportError("Unable to load DLSS 5 runtime.py")
        runtime = importlib.util.module_from_spec(runtime_spec)
        sys.modules[runtime_name] = runtime
        runtime_spec.loader.exec_module(runtime)
    if video is None:
        video_spec = importlib.util.spec_from_file_location(
            video_name, package_dir / "video.py"
        )
        if video_spec is None or video_spec.loader is None:
            raise ImportError("Unable to load DLSS 5 video.py")
        video = importlib.util.module_from_spec(video_spec)
        sys.modules[video_name] = video
        video_spec.loader.exec_module(video)
    return runtime, video


def _to_rgba(images: torch.Tensor) -> np.ndarray:
    if images.ndim != 4 or images.shape[-1] not in (3, 4):
        raise ValueError("ComfyUI IMAGE must have shape [B,H,W,3] or [B,H,W,4].")
    pixels = (images.detach().cpu().float().clamp(0, 1).numpy() * 255.0).round()
    pixels = pixels.astype(np.uint8, copy=False)
    if pixels.shape[-1] == 3:
        alpha = np.full((*pixels.shape[:3], 1), 255, dtype=np.uint8)
        pixels = np.concatenate((pixels, alpha), axis=-1)
    return np.ascontiguousarray(pixels)


class AerinceDLSS5Image:
    """Apply the local DLSS 5 feature-18 image pipeline to each input image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "输入图像": ("IMAGE",),
                "神经渲染预设": (["默认", "预设 #1", "预设 #2", "预设 #3"],),
                "神经渲染风格": (["默认", "自然", "电影感"],),
                "放大倍率": (["1.0（原尺寸）", "1.5（质量）", "1.724（平衡）", "2.0（性能）", "3.0（超高性能）"],),
                "神经渲染强度": ("FLOAT", {"default": 1.10, "min": 0.0, "max": 2.0, "step": 0.05}),
                "局部色调强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "局部结构强度": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "皮肤结构强度": ("FLOAT", {"default": -1.0, "min": -1.0, "max": 2.0, "step": 0.05}),
                "自动蒙版": (["关闭", "开启"],),
                "DLSS模型预设": (["默认", "J", "K", "L", "M"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "apply"
    CATEGORY = "Aerince Tools/Image"

    def apply(
        self,
        输入图像: torch.Tensor,
        神经渲染预设: str,
        神经渲染风格: str,
        放大倍率: str,
        神经渲染强度: float,
        局部色调强度: float,
        局部结构强度: float,
        皮肤结构强度: float,
        自动蒙版: str,
        DLSS模型预设: str,
    ):
        nr_preset = {"默认": "Default", "预设 #1": "Preset #1", "预设 #2": "Preset #2", "预设 #3": "Preset #3"}[神经渲染预设]
        nr_style = {"默认": "Default", "自然": "Natural", "电影感": "Cinematic"}[神经渲染风格]
        upscaling_factor = {"1.0（原尺寸）": "1.0", "1.5（质量）": "1.5", "1.724（平衡）": "1.724", "2.0（性能）": "2.0", "3.0（超高性能）": "3.0"}[放大倍率]
        dlss_model_preset = DLSS模型预设
        if dlss_model_preset == "默认":
            dlss_model_preset = "Default"
        images = 输入图像
        nr_intensity = 神经渲染强度
        local_tone_strength = 局部色调强度
        local_structure_strength = 局部结构强度
        skin_structure_strength = 皮肤结构强度
        automatic_mask = 自动蒙版
        if images.shape[1] < 64 or images.shape[2] < 64:
            raise ValueError("DLSS 5 requires both input dimensions to be at least 64 pixels.")

        runtime, video = _load_runtime_modules()
        options = video.ConversionOptions(
            nr_preset=nr_preset,
            nr_style=nr_style,
            nr_intensity=nr_intensity,
            local_tone_strength=local_tone_strength,
            local_structure_strength=local_structure_strength,
            skin_structure_strength=skin_structure_strength,
            automatic_mask=automatic_mask == "开启",
            upscaling_factor=float(upscaling_factor),
            dlss_model_preset=dlss_model_preset,
        )
        factor, mode = video.resolve_upscaling_mode(options.upscaling_factor)
        native_settings = video.resolve_native_settings(options)
        rgba = _to_rgba(images)
        input_height, input_width = rgba.shape[1:3]
        output_width, output_height = video.resolve_output_size(
            input_width, input_height, factor
        )

        runtime.validate_runtime_files()
        gpu = runtime.detect_gpu()
        bundle = runtime.inspect_runtime_bundle()
        runtime.validate_gpu_runtime(gpu, bundle)

        output = np.empty(
            (rgba.shape[0], output_height, output_width, 4), dtype=np.uint8
        )
        with _DLSS_LOCK:
            with runtime.active_job() as controller:
                session = runtime.DLSSFrameSession(
                    input_width=input_width,
                    input_height=input_height,
                    output_width=output_width,
                    output_height=output_height,
                    frame_count=rgba.shape[0],
                    warmup_frames=0,
                    factor=factor,
                    mode=mode,
                    native_settings=native_settings,
                    gpu=gpu,
                    runtime_bundle=bundle,
                    controller=controller,
                )
                try:
                    motion = np.zeros(
                        (session.render_height, session.render_width, 2),
                        dtype=np.float16,
                    )
                    for index, frame in enumerate(rgba):
                        fitted = runtime.resize_fit(
                            frame, session.render_width, session.render_height
                        )
                        processed, _ = session.process(
                            index=index,
                            rgba=fitted,
                            motion=motion,
                            reset=True,
                            pts=index,
                        )
                        alpha = frame[..., 3]
                        if alpha.shape != (output_height, output_width):
                            alpha = cv2.resize(
                                alpha,
                                (output_width, output_height),
                                interpolation=cv2.INTER_LANCZOS4,
                            )
                        processed[..., 3] = alpha
                        output[index] = processed
                finally:
                    session.close()

        if not output[..., :3].any():
            raise RuntimeError(
                "DLSS 5 returned an all-black image. Use an input at least 128×128 "
                "and check ReShade.log for feature-18 diagnostics."
            )
        result = torch.from_numpy(output).to(device=images.device, dtype=images.dtype) / 255.0
        return (result[..., :3],)


NODE_CLASS_MAPPINGS = {"AerinceDLSS5Image": AerinceDLSS5Image}
NODE_DISPLAY_NAME_MAPPINGS = {
    "AerinceDLSS5Image": "Aerince DLSS 5 Image Enhancer",
}
