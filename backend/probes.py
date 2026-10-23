from __future__ import annotations

import base64
import math
import time
from typing import Any

import cv2
import numpy as np
import torch
import torch.nn.functional as F


VIEW_RAMP = np.array(
    [
        [4, 9, 18],
        [10, 42, 68],
        [8, 120, 150],
        [94, 231, 255],
        [240, 252, 255],
    ],
    dtype=np.float32,
)

STATIONS = [
    {"id": 0, "name": "找边缘", "en": "EDGES", "range": "0-2"},
    {"id": 1, "name": "认形状", "en": "SHAPES", "range": "3-10"},
    {"id": 2, "name": "汇合尺度", "en": "FUSION", "range": "11-22"},
    {"id": 3, "name": "下结论", "en": "DECIDE", "range": "23"},
]

TYPE_CAPTIONS = {
    "Conv": "用卷积窗口在图上寻找局部图案",
    "C3k2": "把通道分流提取，再合并回主路",
    "SPPF": "用多个大小的窗口一起看同一张图",
    "C2PSA": "把注意力放到更重要的位置",
    "Upsample": "把特征图放大，准备和浅层信息汇合",
    "Concat": "把不同深度的特征并排拼在一起",
    "Detect": "一次给出最终答案，不需要 NMS",
    "Pose26": "一次给出最终答案：框、置信度和 17 个关键点",
}

TYPE_DETAILS = {
    "Conv": "卷积核在局部窗口内做乘加，输出新的特征图。",
    "C3k2": "跨阶段部分网络，把特征拆成几路处理，减少重复计算。",
    "SPPF": "串联不同尺度的最大池化，让同一层同时拥有大小视野。",
    "C2PSA": "位置敏感注意力，让网络更关注画面里的关键区域。",
    "Upsample": "最近邻上采样，把低分辨率特征图放大。",
    "Concat": "沿通道维拼接两路特征，不改变空间尺寸。",
    "Detect": "YOLO26 端到端检测头：300 个查询槽位直接输出框和置信度。",
    "Pose26": "YOLO26 端到端姿态头：300 个查询槽位直接输出框、置信度、类别和 17 个关键点。",
}


def _first_conv(module: torch.nn.Module) -> torch.nn.Conv2d | None:
    if hasattr(module, "conv") and isinstance(module.conv, torch.nn.Conv2d):
        return module.conv
    for sub in module.modules():
        if isinstance(sub, torch.nn.Conv2d):
            return sub
    return None


def _shape_of(output: Any) -> list[int] | None:
    tensor = output[0] if isinstance(output, (tuple, list)) else output
    if torch.is_tensor(tensor):
        return [int(x) for x in tensor.shape]
    return None


class ProbeTaps:
    """Captures real YOLO26 intermediate maps, conv slices and raw head output."""

    def __init__(
        self,
        torch_model: torch.nn.Module,
        device: torch.device,
        input_size: int = 640,
        map_size: int = 128,
    ):
        self.torch_model = torch_model
        self.device = device
        self.input_size = input_size
        self.map_size = map_size
        self.layers = list(torch_model.model)
        self.layer_count = len(self.layers)
        self.capture_maps = False
        self.capture_micro = False
        self.map_layers: set[int] | None = None
        self.micro_layers: set[int] | None = None
        self.roi_norm: tuple[float, float, float, float] | None = None
        self._pending_maps: dict[int, tuple[torch.Tensor, list[int]]] = {}
        self._pending_head: np.ndarray | None = None
        self._micro: dict[int, dict[str, Any]] = {}
        self.latest_maps: list[dict[str, Any]] = []
        self.latest_head: dict[str, Any] | None = None
        self.latest_micro: dict[int, dict[str, Any]] = {}
        self.shapes: dict[int, list[int]] = {}
        self.times: dict[int, float] = {}
        self.meta: list[dict[str, Any]] = []
        self.total_params = 0
        self._survey()
        self._build_meta()
        self._install_hooks()

    def _survey(self) -> None:
        """One-time shape + GPU timing pass, before permanent hooks are installed."""
        self.torch_model.eval()
        dummy = torch.rand(1, 3, self.input_size, self.input_size, device=self.device)
        is_cuda = self.device.type == "cuda"
        events: dict[int, tuple[Any, Any]] = {}
        cpu_starts: dict[int, float] = {}
        handles = []

        def pre_hook(module, args, index: int):
            if is_cuda:
                start, _ = events[index]
                start.record()
            else:
                cpu_starts[index] = time.perf_counter()

        def post_hook(module, args, output, index: int):
            if is_cuda:
                _, end = events[index]
                end.record()
            else:
                self.times[index] = (time.perf_counter() - cpu_starts.get(index, time.perf_counter())) * 1000
            self.shapes[index] = _shape_of(output) or []

        try:
            with torch.inference_mode():
                self.torch_model(dummy)
                for index, layer in enumerate(self.layers):
                    if is_cuda:
                        events[index] = (
                            torch.cuda.Event(enable_timing=True),
                            torch.cuda.Event(enable_timing=True),
                        )
                    handles.append(layer.register_forward_pre_hook(lambda m, a, i=index: pre_hook(m, a, i)))
                    handles.append(layer.register_forward_hook(lambda m, a, o, i=index: post_hook(m, a, o, i)))
                self.torch_model(dummy)
            if is_cuda:
                torch.cuda.synchronize()
                for index, (start, end) in events.items():
                    self.times[index] = float(start.elapsed_time(end))
        except Exception as exc:
            print(f"[probe] survey skipped: {exc}")
        finally:
            for handle in handles:
                handle.remove()

    def _build_meta(self) -> None:
        total = 0
        for index, layer in enumerate(self.layers):
            params = sum(p.numel() for p in layer.parameters())
            total += params
            type_name = type(layer).__name__
            conv = _first_conv(layer)
            group = "BACKBONE" if index <= 10 else "NECK" if index <= 22 else "HEAD"
            station = 0 if index <= 2 else 1 if index <= 10 else 2 if index <= 22 else 3
            caption = TYPE_CAPTIONS.get(type_name, f"{type_name} 层")
            detail = TYPE_DETAILS.get(type_name, "YOLO26 中间特征处理层。")
            if type_name == "Conv" and conv is not None:
                caption = f"{conv.kernel_size[0]}×{conv.kernel_size[0]} 卷积 · 寻找局部图案"
            if type_name == "C3k2":
                caption = "C3k2 · 分流提取后合回主路"
            if index == 23:
                caption = "Pose26 端到端头 · 300 槽位给出框和关键点"
            self.meta.append(
                {
                    "i": index,
                    "name": f"{index:02d}",
                    "type": type_name,
                    "group": group,
                    "station": station,
                    "params": int(params),
                    "shape": self.shapes.get(index),
                    "kernel": int(conv.kernel_size[0]) if conv is not None else None,
                    "stride": int(conv.stride[0]) if conv is not None else None,
                    "padding": int(conv.padding[0]) if conv is not None else None,
                    "groups": int(conv.groups) if conv is not None else None,
                    "is_conv": type_name == "Conv",
                    "caption_zh": caption,
                    "detail_zh": detail,
                    "gpu_ms": round(float(self.times.get(index, 0.0)), 3) if self.times else None,
                }
            )
        self.total_params = total

    def _install_hooks(self) -> None:
        for index, layer in enumerate(self.layers):
            layer.register_forward_hook(self._map_hook(index))
            if isinstance(layer, torch.nn.Module) and hasattr(layer, "conv") and isinstance(layer.conv, torch.nn.Conv2d):
                layer.register_forward_pre_hook(self._conv_pre_hook(index))
                layer.register_forward_hook(self._conv_post_hook(index))

    def begin_probe(self, roi_norm: tuple[float, float, float, float] | None) -> None:
        self.roi_norm = roi_norm
        self._pending_maps.clear()
        self._pending_head = None
        self._micro.clear()
        self.capture_maps = True
        self.capture_micro = True

    def _map_hook(self, index: int):
        def hook(module, args, output):
            if not self.capture_maps:
                return
            if self.map_layers is not None and index not in self.map_layers:
                return
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(tensor):
                return
            if tensor.ndim == 4:
                small = self._reduce_map(tensor)
                if small is not None:
                    self._pending_maps[index] = (small, [int(x) for x in tensor.shape])
            elif tensor.ndim == 3 and index == self.layer_count - 1:
                self._pending_head = tensor.detach().float().cpu().numpy()

        return hook

    def _reduce_map(self, tensor: torch.Tensor):
        try:
            x = tensor.detach().float()
            if x.shape[0] != 1:
                x = x[:1]
            channels = x.shape[1]
            if channels <= 3:
                selected = x[0]
                if channels == 1:
                    selected = selected.repeat(3, 1, 1)
                elif channels == 2:
                    selected = torch.cat([selected, selected[:1]], dim=0)
            else:
                variance = x[0].var(dim=(1, 2), unbiased=False)
                count = min(3, channels)
                indices = torch.topk(variance, count).indices
                selected = x[0][indices]
                if count == 1:
                    selected = selected.repeat(3, 1, 1)
                elif count == 2:
                    selected = torch.cat([selected, selected[:1]], dim=0)
            small = F.interpolate(
                selected.unsqueeze(0),
                size=(self.map_size, self.map_size),
                mode="bilinear",
                align_corners=False,
            )[0]
            return small
        except Exception:
            return None

    def _conv_pre_hook(self, index: int):
        def hook(module, args):
            if not self.capture_micro or not args:
                return
            if self.micro_layers is not None and index not in self.micro_layers:
                return
            tensor = args[0]
            if not torch.is_tensor(tensor) or tensor.ndim != 4:
                return
            self._micro[index] = {"input": tensor}

        return hook

    def _conv_post_hook(self, index: int):
        def hook(module, args, output):
            entry = self._micro.get(index)
            if not self.capture_micro or entry is None or "input" not in entry:
                return
            try:
                input_tensor = entry.pop("input")
                output_tensor = output[0] if isinstance(output, (tuple, list)) else output
                if not torch.is_tensor(input_tensor) or input_tensor.ndim != 4:
                    return
                conv = getattr(module, "conv", None)
                if not isinstance(conv, torch.nn.Conv2d) or conv.groups != 1:
                    return
                x = input_tensor.detach().float()
                y = output_tensor.detach().float() if torch.is_tensor(output_tensor) else None
                if y is None or y.ndim != 4:
                    return
                roi = self.roi_norm or (0.34, 0.28, 0.66, 0.72)
                _, channels, height, width = x.shape
                _, out_channels, out_h, out_w = y.shape
                x1 = max(0, min(width - 1, int(roi[0] * width)))
                y1 = max(0, min(height - 1, int(roi[1] * height)))
                x2 = max(x1 + 1, min(width, int(roi[2] * width)))
                y2 = max(y1 + 1, min(height, int(roi[3] * height)))
                channel_var = x[0, :, y1:y2, x1:x2].var(dim=(1, 2), unbiased=False)
                channel_index = torch.argmax(channel_var)
                ox1 = max(0, min(out_w - 1, int(roi[0] * out_w)))
                oy1 = max(0, min(out_h - 1, int(roi[1] * out_h)))
                ox2 = max(ox1 + 1, min(out_w, int(roi[2] * out_w)))
                oy2 = max(oy1 + 1, min(out_h, int(roi[3] * out_h)))
                out_activity = y[0, :, oy1:oy2, ox1:ox2].abs().mean(dim=(1, 2))
                out_index = torch.argmax(out_activity)

                kernel = int(conv.kernel_size[0])
                stride = int(conv.stride[0])
                patch_size = kernel + 2 * stride
                patch_size = max(kernel, min(patch_size, height, width))
                center_x = int(round((x1 + x2) * 0.5 - patch_size * 0.5))
                center_y = int(round((y1 + y2) * 0.5 - patch_size * 0.5))
                start_x = max(0, min(width - patch_size, center_x))
                start_y = max(0, min(height - patch_size, center_y))
                patch = x[0, channel_index, start_y : start_y + patch_size, start_x : start_x + patch_size]
                context_size = min(max(patch_size * 3, 21), width, height)
                context_x = max(0, min(width - context_size, start_x + patch_size // 2 - context_size // 2))
                context_y = max(0, min(height - context_size, start_y + patch_size // 2 - context_size // 2))
                context = x[
                    0,
                    channel_index,
                    context_y : context_y + context_size,
                    context_x : context_x + context_size,
                ]
                weight = conv.weight[out_index, channel_index]
                entry.update(
                    {
                        "patch": patch.detach().clone(),
                        "context": context.detach().float().clone(),
                        "context_origin": (context_x, context_y),
                        "context_size": int(context_size),
                        "patch_origin": (start_x, start_y),
                        "weight": weight.detach().float().clone(),
                        "channel_in": channel_index.detach(),
                        "channel_out": out_index.detach(),
                        "kernel": kernel,
                        "stride": stride,
                        "patch_size": int(patch_size),
                        "channels_in": int(channels),
                        "channels_out": int(out_channels),
                        "spatial": [int(width), int(height)],
                    }
                )
            except Exception:
                self._micro.pop(index, None)

        return hook

    def finish_probe(self) -> dict[str, Any]:
        self.capture_maps = False
        self.capture_micro = False
        pending = [
            (index, self._pending_maps[index])
            for index in range(self.layer_count)
            if index in self._pending_maps
        ]
        images: dict[int, np.ndarray] = {}
        shapes: dict[int, list[int]] = {}
        if pending:
            try:
                # 23 张小图一起归一化再拷回 CPU，省掉逐图同步。
                stack = torch.stack([small.float() for _, (small, _) in pending])
                low = stack.amin(dim=(2, 3), keepdim=True)
                high = stack.amax(dim=(2, 3), keepdim=True)
                stack = (stack - low) / (high - low + 1e-6)
                stack = torch.pow(stack.clamp_(0, 1), 0.85)
                batch = (stack * 255).to(torch.uint8).cpu().numpy()
                for row, (index, (_, shape)) in zip(batch, pending):
                    images[index] = row
                    shapes[index] = shape
            except Exception:
                images.clear()
                shapes.clear()
                pending = []

        maps: list[dict[str, Any]] = []
        for index in range(self.layer_count):
            entry: dict[str, Any] = {"i": index, "map": None}
            image = images.get(index)
            if image is not None:
                try:
                    image = image.transpose(1, 2, 0)
                    entry["activity"] = round(float(image.mean()) / 255.0, 4)
                    rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    ok, buffer = cv2.imencode(
                        ".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 68]
                    )
                    if ok:
                        entry["map"] = base64.b64encode(buffer.tobytes()).decode("ascii")
                except Exception:
                    entry["map"] = None
                entry["shape"] = shapes[index]
            maps.append(entry)

        head = self._head_payload(self._pending_head)
        micro = self._finish_micro()
        self.latest_maps = maps
        self.latest_head = head
        self.latest_micro = micro
        self._pending_maps.clear()
        self._pending_head = None
        self._micro.clear()
        return {"maps": maps, "head": head, "micro": micro}

    def _finish_micro(self) -> dict[int, dict[str, Any]]:
        micro: dict[int, dict[str, Any]] = {}
        for index, entry in self._micro.items():
            patch_tensor = entry.get("patch")
            weight_tensor = entry.get("weight")
            if patch_tensor is None or weight_tensor is None:
                continue
            try:
                patch = patch_tensor.cpu().numpy()
                weight = weight_tensor.cpu().numpy()
                kernel = int(entry["kernel"])
                stride = int(entry["stride"])
                patch_size = int(entry["patch_size"])
                positions = list(range(0, max(0, patch_size - kernel) + 1, max(1, stride)))
                if not positions:
                    positions = [0]
                steps = []
                for py in positions:
                    for px in positions:
                        window = patch[py : py + kernel, px : px + kernel]
                        if window.shape != weight.shape:
                            continue
                        products = (window * weight).reshape(-1)
                        steps.append(
                            {
                                "x": int(px),
                                "y": int(py),
                                "products": [round(float(v), 3) for v in products],
                                "sum": round(float(products.sum()), 3),
                            }
                        )
                view = None
                context_tensor = entry.get("context")
                if context_tensor is not None:
                    try:
                        ctx = context_tensor.cpu().numpy().astype(np.float32)
                        low = float(ctx.min())
                        high = float(ctx.max())
                        ctx = (ctx - low) / (high - low + 1e-6)
                        ctx = np.power(np.clip(ctx, 0.0, 1.0), 0.65)
                        stops = np.linspace(0, 255, len(VIEW_RAMP))
                        ramp = np.stack(
                            [
                                np.interp(np.arange(256), stops, VIEW_RAMP[:, channel])
                                for channel in range(3)
                            ],
                            axis=1,
                        ).astype(np.uint8)
                        image = ramp[(ctx * 255).astype(np.uint8)]
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        image = cv2.resize(image, (168, 168), interpolation=cv2.INTER_NEAREST)
                        ok, buffer = cv2.imencode(
                            ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), 72]
                        )
                        if ok:
                            context_size = int(entry["context_size"])
                            patch_x, patch_y = entry["patch_origin"]
                            context_x, context_y = entry["context_origin"]
                            view = {
                                "image": base64.b64encode(buffer.tobytes()).decode("ascii"),
                                "patch": [
                                    round((patch_x - context_x) / context_size, 4),
                                    round((patch_y - context_y) / context_size, 4),
                                    round(patch_size / context_size, 4),
                                    round(patch_size / context_size, 4),
                                ],
                                "context": context_size,
                                "source": entry["spatial"],
                            }
                    except Exception:
                        view = None
                micro[index] = {
                    "layer": index,
                    "patch": [[round(float(v), 3) for v in row] for row in patch],
                    "weights": [[round(float(v), 3) for v in row] for row in weight],
                    "steps": steps,
                    "view": view,
                    "kernel": kernel,
                    "stride": stride,
                    "patch_size": patch_size,
                    "channel_in": int(entry["channel_in"].item()),
                    "channel_out": int(entry["channel_out"].item()),
                    "channels_in": int(entry["channels_in"]),
                    "channels_out": int(entry["channels_out"]),
                    "spatial": entry["spatial"],
                }
            except Exception:
                continue
        return micro

    def _head_payload(self, raw: np.ndarray | None) -> dict[str, Any] | None:
        if raw is None:
            return None
        try:
            data = raw[0] if raw.ndim == 3 else raw
            if data.ndim != 2 or data.shape[1] < 6:
                return None
            scores = data[:, 4].astype(np.float32)
            order = np.argsort(scores)[::-1]
            top = []
            for slot in order[:8]:
                score = float(scores[slot])
                if score < 0.02:
                    continue
                box = data[slot, :4].astype(np.float32) / float(self.input_size)
                top.append(
                    {
                        "slot": int(slot),
                        "score": round(score, 4),
                        "bbox": [round(float(v), 4) for v in np.clip(box, 0, 1)],
                    }
                )
            return {
                "slots": [round(float(v), 3) for v in scores.tolist()],
                "top": top,
                "active": int((scores > 0.25).sum()),
                "queries": int(data.shape[0]),
            }
        except Exception:
            return None
