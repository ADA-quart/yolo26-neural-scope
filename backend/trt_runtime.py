"""TensorRT fast path for YOLO26 pose. Probe frames stay on PyTorch."""

from __future__ import annotations

import functools
import hashlib
import os
import re
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
TRT_DIR = ROOT / "state" / "trt"
INPUT_SHAPE = (1, 3, 640, 640)
_DLL_HANDLES: list[Any] = []


def _add_dll_dirs() -> None:
    if os.name != "nt":
        return
    torch_lib = Path(torch.__file__).resolve().parent / "lib"
    if torch_lib.is_dir():
        try:
            _DLL_HANDLES.append(os.add_dll_directory(str(torch_lib)))
        except OSError:
            pass


def load_trt():
    _add_dll_dirs()
    try:
        import tensorrt as trt  # type: ignore

        return trt
    except ImportError:
        import tensorrt_bindings as trt  # type: ignore

        return trt


def _device_capability() -> tuple[int, int]:
    if not torch.cuda.is_available():
        return (0, 0)
    major, minor = torch.cuda.get_device_capability(0)
    return int(major), int(minor)


def gpu_slug() -> str:
    name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    major, minor = _device_capability()
    return f"{slug}-sm{major}{minor}" if major else slug


@functools.lru_cache(maxsize=8)
def weights_sha1(path: str) -> str:
    digest = hashlib.sha1()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()[:12]


def preferred_precisions() -> list[str]:
    # 图灵及更早默认 FP32；Ampere 起默认 FP16。
    return ["fp16", "fp32"] if _device_capability()[0] >= 8 else ["fp32"]


def engine_name(weights: Path, precision: str, trt_version: str) -> str:
    return (
        f"{weights.stem}_640_{precision}_{gpu_slug()}"
        f"_trt{trt_version}_{weights_sha1(str(weights))}.engine"
    )


def engine_path(weights: Path, precision: str, trt_version: str) -> Path:
    return TRT_DIR / engine_name(weights, precision, trt_version)


def find_engine(weights: Path) -> tuple[Path, str] | None:
    trt = load_trt()
    version = str(getattr(trt, "__version__", "0"))
    ordered = list(dict.fromkeys(preferred_precisions() + ["fp16", "fp32"]))
    for precision in ordered:
        candidate = engine_path(weights, precision, version)
        if candidate.is_file() and candidate.stat().st_size > 0:
            return candidate, precision
    return None


class TensorRTRunner:
    """Static-shape FP32 engine using PyTorch's current CUDA stream."""

    def __init__(self, path: Path):
        self.trt = load_trt()
        self.trt_version = str(getattr(self.trt, "__version__", "?"))
        self.path = Path(path)
        logger = self.trt.Logger(self.trt.Logger.WARNING)
        self._runtime = self.trt.Runtime(logger)
        self.engine = self._runtime.deserialize_cuda_engine(self.path.read_bytes())
        if self.engine is None:
            raise RuntimeError(f"无法反序列化 TensorRT 引擎：{self.path.name}")
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("无法创建 TensorRT 执行上下文")

        self.input_name = ""
        self.output_name = ""
        for index in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(index)
            mode = self.engine.get_tensor_mode(name)
            shape = tuple(int(v) for v in self.engine.get_tensor_shape(name))
            dtype = self.engine.get_tensor_dtype(name)
            if mode == self.trt.TensorIOMode.INPUT:
                if shape != INPUT_SHAPE:
                    raise RuntimeError(f"引擎输入形状 {shape} 不是 {INPUT_SHAPE}")
                if dtype != self.trt.DataType.FLOAT:
                    raise RuntimeError("引擎输入不是 FP32")
                self.input_name = name
            else:
                if dtype != self.trt.DataType.FLOAT:
                    raise RuntimeError("引擎输出不是 FP32")
                self.output_name = name
                self.output_shape = shape

        if not self.input_name or not self.output_name:
            raise RuntimeError("引擎缺少输入或输出张量")

        self._output = torch.empty(self.output_shape, dtype=torch.float32, device="cuda")

    def run(self, tensor: torch.Tensor) -> torch.Tensor:
        x = tensor.contiguous()
        if tuple(x.shape) != INPUT_SHAPE:
            raise ValueError(f"输入形状 {tuple(x.shape)} 不是 {INPUT_SHAPE}")
        stream = torch.cuda.current_stream(x.device).cuda_stream
        self.context.set_tensor_address(self.input_name, int(x.data_ptr()))
        self.context.set_tensor_address(self.output_name, int(self._output.data_ptr()))
        if not self.context.execute_async_v3(stream):
            raise RuntimeError("TensorRT execute_async_v3 返回失败")
        return self._output
