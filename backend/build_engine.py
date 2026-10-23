"""Build a TensorRT engine for the current GPU.

Usage:
    python backend/build_engine.py          # V100 -> FP32, Ada+ -> FP16
    python backend/build_engine.py --fp32
    python backend/build_engine.py --half --force
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from trt_runtime import (  # noqa: E402
    INPUT_SHAPE,
    ROOT,
    TRT_DIR,
    TensorRTRunner,
    engine_path,
    gpu_slug,
    load_trt,
    preferred_precisions,
    weights_sha1,
)

WEIGHTS = ROOT / "weights" / "yolo26n-pose.pt"


def export_onnx(weights: Path, digest: str, force: bool) -> Path:
    target = TRT_DIR / f"{weights.stem}_640_{digest}.onnx"
    if target.is_file() and not force:
        print(f"[onnx] 复用 {target.name}")
        return target

    from ultralytics import YOLO

    TRT_DIR.mkdir(parents=True, exist_ok=True)
    temp_weights = TRT_DIR / weights.name
    shutil.copy2(weights, temp_weights)
    try:
        print("[onnx] 导出 ONNX…", flush=True)
        exported = YOLO(str(temp_weights)).export(
            format="onnx",
            imgsz=640,
            batch=1,
            simplify=False,
            opset=17,
            device=0,
        )
        onnx_file = Path(str(exported))
        if not onnx_file.is_file():
            raise RuntimeError(f"Ultralytics 未生成 ONNX：{exported}")
        target.unlink(missing_ok=True)
        shutil.move(str(onnx_file), target)
    finally:
        temp_weights.unlink(missing_ok=True)
    print(f"[onnx] {target} ({target.stat().st_size / 1024 ** 2:.1f} MB)")
    return target


def build(onnx_file: Path, out_file: Path, precision: str) -> float:
    trt = load_trt()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    parser = trt.OnnxParser(network, logger)
    # TensorRT 的 C++ 解析器在 Windows 上打不开含中文的路径，直接喂字节。
    if not parser.parse(onnx_file.read_bytes()):
        errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
        raise RuntimeError("ONNX 解析失败：" + (" | ".join(errors[:3]) or "未知错误"))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
    if precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("当前 GPU 不支持快速 FP16，请改用 --fp32")
        config.set_flag(trt.BuilderFlag.FP16)

    print(f"[trt] 构建 {precision.upper()} 引擎，约 2–3 分钟…", flush=True)
    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("TensorRT 构建失败，请查看上方日志")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    temp_file = out_file.with_name(out_file.name + ".tmp")
    with open(temp_file, "wb") as handle:
        handle.write(serialized)
    temp_file.replace(out_file)
    return time.perf_counter() - started


def main() -> int:
    parser = argparse.ArgumentParser(description="为当前 GPU 构建 YOLO26n-Pose TensorRT 引擎")
    parser.add_argument("--half", action="store_true", help="强制 FP16")
    parser.add_argument("--fp32", action="store_true", help="强制 FP32")
    parser.add_argument("--force", action="store_true", help="覆盖已有引擎并重新导出 ONNX")
    args = parser.parse_args()
    if args.half and args.fp32:
        parser.error("--half 与 --fp32 不能同时使用")
    if not torch.cuda.is_available():
        print("错误：构建 TensorRT 引擎需要 NVIDIA GPU", file=sys.stderr)
        return 1
    if not WEIGHTS.is_file():
        print(f"错误：找不到权重 {WEIGHTS}", file=sys.stderr)
        return 1

    precision = "fp16" if args.half else "fp32" if args.fp32 else preferred_precisions()[0]
    trt = load_trt()
    version = str(getattr(trt, "__version__", "0"))
    digest = weights_sha1(str(WEIGHTS))
    out_file = engine_path(WEIGHTS, precision, version)
    print(f"[gpu] {torch.cuda.get_device_name(0)} · {gpu_slug()}")
    print(f"[trt] TensorRT {version} · {precision.upper()} · 权重 {digest}")

    if out_file.is_file() and not args.force:
        print(f"[skip] 引擎已存在：{out_file}")
        return 0

    onnx_file = export_onnx(WEIGHTS, digest, args.force)
    seconds = build(onnx_file, out_file, precision)
    TensorRTRunner(out_file)

    manifest = {
        "engine": out_file.name,
        "precision": precision,
        "gpu": torch.cuda.get_device_name(0),
        "cc": list(torch.cuda.get_device_capability(0)),
        "trt_version": version,
        "weights": WEIGHTS.name,
        "weights_sha1": digest,
        "onnx": onnx_file.name,
        "input_shape": list(INPUT_SHAPE),
        "build_seconds": round(seconds, 1),
        "engine_mb": round(out_file.stat().st_size / 1024 ** 2, 1),
        "built_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    out_file.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {out_file} ({manifest['engine_mb']} MB, {seconds:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
