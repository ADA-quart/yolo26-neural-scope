from __future__ import annotations

import base64
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from identity import FaceEmbedder, FaceGallery, align_face_from_pose, face_box_from_pose
from probes import ProbeTaps
from tracker import FaceTrack, FaceTracker
from trt_runtime import TensorRTRunner, find_engine

ROOT = Path(__file__).resolve().parents[1]
IMG_SIZE = 640
PERSON_CONF = 0.25
MATCH_THRESHOLD = 0.45
# 0 表示后台探针完成即接下一帧；单次前向约 60-100ms，因此实际受算力限流。
PROBE_INTERVAL = 0.0
# 没有后台探针线程时退回内联探针，必须留出间隔，否则会拖垮实时帧率。
PROBE_FALLBACK_INTERVAL = 0.3
DISPLAY_WIDTH = 960
DISPLAY_HEIGHT = 960

DEMO_SCENES: dict[str, dict[str, Any]] = {
    "closeup": {"start": 418, "end": 522, "label": "示例视频 · 双人近景"},
    "multi": {"start": 660, "end": 840, "label": "示例视频 · 三人街景"},
}


@dataclass
class ProbeRequest:
    generation: int
    seq: int
    t: float
    tensor: torch.Tensor
    roi: tuple[float, float, float, float] | None
    live: dict[str, Any]
    face: dict[str, Any] | None
    selected_id: int | None
    ready: torch.cuda.Event


@dataclass
class SourceSpec:
    kind: str = "demo"
    value: str = "closeup"
    label: str = "示例视频 · 双人近景"
    loop: bool = True
    start: int = 418
    end: int = 522

    def public(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "value": self.value,
            "label": self.label,
            "loop": self.loop,
            "start": self.start,
            "end": self.end,
        }


class Engine:
    """Single inference worker. Reads video, runs YOLO26 pose, tracks faces, taps probes."""

    def __init__(self, emit: Callable[[dict[str, Any]], None] | None = None):
        self.emit = emit or (lambda _msg: None)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            # 每帧只有很小的 CPU 预处理，16 线程池会在帧间空转抢满 CPU。
            torch.set_num_threads(2)
            cv2.setNumThreads(2)

        weights = ROOT / "weights" / "yolo26n-pose.pt"
        self.yolo = YOLO(str(weights))
        self.torch_model = self.yolo.model.to(self.device).eval()
        # 融合 Conv+BN 并丢弃推理用不到的 one2many 分支，前向约快 35%。
        self.torch_model.fuse()
        self.taps = ProbeTaps(self.torch_model, self.device, input_size=IMG_SIZE)

        self.trt: TensorRTRunner | None = None
        self._runtime_label = "PyTorch"
        self._engine_name: str | None = None
        self._last_runtime = "pytorch"
        self._probe_async = False
        self._probe_thread: threading.Thread | None = None
        self._probe_requests: queue.Queue[ProbeRequest | None] = queue.Queue(maxsize=1)
        self._probe_stop = threading.Event()
        self._probe_generation = 0
        self._probe_stream: torch.cuda.Stream | None = None
        self._probe_lock = threading.RLock()
        self._init_trt(weights)

        self.embedder = FaceEmbedder(ROOT / "weights" / "edgeface_s_gamma_05.onnx")
        self.gallery = FaceGallery(ROOT / "state", self.embedder)
        demo_dir = ROOT / "assets" / "demo_faces"
        if demo_dir.exists():
            self.gallery.ensure_demo_many(
                [
                    (demo_dir / "a.jpg", "示例人物 A"),
                    (demo_dir / "b.jpg", "示例人物 B"),
                    (demo_dir / "c.jpg", "示例人物 C"),
                    (demo_dir / "d.jpg", "示例人物 D"),
                ]
            )
        else:
            demo_face = ROOT / "assets" / "demo_face.jpg"
            if demo_face.exists():
                self.gallery.ensure_demo(demo_face)

        self.tracker = FaceTracker()
        scene = DEMO_SCENES["closeup"]
        self.source = SourceSpec(
            kind="demo",
            value="closeup",
            label=scene["label"],
            start=scene["start"],
            end=scene["end"],
        )
        self._cap: cv2.VideoCapture | None = None
        self._src_fps = 12.0
        self._next_due = 0.0
        self._frame_no = 0
        self._paused = False
        self._step_once = False
        self._selected_id: int | None = None
        self._selected_aligned: np.ndarray | None = None
        self._selected_owner: int | None = None
        self._crop_lock = threading.Lock()
        self._commands: queue.Queue[dict[str, Any]] = queue.Queue()
        self._stop = threading.Event()
        self.thread: threading.Thread | None = None

        self._fps_marks: list[float] = []
        self._infer_ms = 0.0
        self._probe_ms = 0.0
        self._visible_faces = 0
        self._last_probe_at = 0.0
        self._last_error = ""
        self._head_live: dict[str, Any] | None = None
        self.latest_live: dict[str, Any] | None = None
        self.latest_probe: dict[str, Any] | None = None
        self._best: dict[str, Any] | None = None
        self._held: dict[str, Any] | None = None
        self._state_dirty = True

        self.model_info = {
            "name": "YOLO26n-Pose",
            "params": int(self.taps.total_params),
            "layers": self.taps.layer_count,
            "input": IMG_SIZE,
            "device": str(self.device),
            "gpu": torch.cuda.get_device_name(0) if self.device.type == "cuda" else "CPU",
            "cuda": torch.version.cuda,
            "torch": torch.__version__,
            "weights": weights.name,
            "embedder": "EdgeFace-S 512D",
            "match_threshold": MATCH_THRESHOLD,
            "probe_interval": PROBE_INTERVAL if self._probe_async else PROBE_FALLBACK_INTERVAL,
            "probe_continuous": self._probe_async and PROBE_INTERVAL <= 0.0,
            "runtime": self._runtime_label,
            "engine": self._engine_name,
        }
        self.layer_meta = self.taps.meta

    def _init_trt(self, weights: Path) -> None:
        if self.device.type != "cuda":
            return
        try:
            found = find_engine(weights)
            if found is None:
                return
            path, precision = found
            runner = TensorRTRunner(path)
        except Exception as exc:
            print(f"[trt] 跳过 TensorRT：{exc}", flush=True)
            return
        self.trt = runner
        self._engine_name = path.name
        self._runtime_label = f"TensorRT {runner.trt_version} · {precision.upper()}"
        print(f"[trt] 快路径已启用：{path.name}", flush=True)
        self._start_probe_worker()

    def _start_probe_worker(self) -> None:
        if self.trt is None or self.device.type != "cuda":
            return
        try:
            self._probe_stream = torch.cuda.Stream(device=self.device)
            self._probe_stop.clear()
            self._probe_async = True
            self._probe_thread = threading.Thread(
                target=self._probe_loop, name="yolo-probe", daemon=True
            )
            self._probe_thread.start()
        except Exception as exc:
            print(f"[probe] 后台探针未启用：{exc}", flush=True)
            self._probe_async = False

    def _shutdown_probe_worker(self) -> None:
        self._probe_async = False
        self._probe_generation += 1
        thread = self._probe_thread
        self._probe_thread = None
        while True:
            try:
                self._probe_requests.get_nowait()
            except queue.Empty:
                break
        if thread is None:
            return
        self._probe_stop.set()
        try:
            self._probe_requests.put_nowait(None)
        except queue.Full:
            pass
        thread.join(timeout=5.0)

    def _probe_loop(self) -> None:
        while not self._probe_stop.is_set():
            try:
                request = self._probe_requests.get(timeout=0.25)
            except queue.Empty:
                continue
            if request is None:
                break
            if request.generation != self._probe_generation:
                continue
            try:
                self._run_probe(request)
            except Exception as exc:
                self.taps.capture_maps = False
                self.taps.capture_micro = False
                self.taps._pending_maps.clear()
                self.taps._pending_head = None
                self.taps._micro.clear()
                self.emit({"type": "error", "message": f"探针失败：{exc}"})

    def _run_probe(self, request: ProbeRequest) -> None:
        stream = self._probe_stream
        if stream is None:
            return
        with torch.cuda.stream(stream):
            request.ready.wait(stream)
            self.taps.begin_probe(request.roi)
            started = time.perf_counter()
            with torch.inference_mode():
                self.torch_model(request.tensor)
            result = self.taps.finish_probe()
            took_ms = (time.perf_counter() - started) * 1000.0
        payload = {
            "type": "probe",
            "seq": request.seq,
            "t": request.t,
            "maps": result["maps"],
            "micro": result["micro"],
            "head": result["head"],
            "roi": request.roi,
            "took_ms": round(took_ms, 1),
            "frame": request.live.get("frame"),
            "faces": request.live.get("faces") or [],
            "face": request.face,
        }
        with self._probe_lock:
            if request.generation != self._probe_generation or self._held is not None:
                return
            self._probe_ms = took_ms
            self._consider_best(request.live, payload, request.selected_id)
            self.latest_probe = payload
            self._state_dirty = True
            self.emit(payload)

    def _submit_probe(
        self,
        tensor: torch.Tensor,
        roi: tuple[float, float, float, float] | None,
        live: dict[str, Any],
        now: float,
    ) -> None:
        if not self._probe_async or self._probe_stream is None:
            return
        try:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            request = ProbeRequest(
                generation=self._probe_generation,
                seq=int(live["seq"]),
                t=float(live["t"]),
                tensor=tensor,
                roi=roi,
                live=live,
                face=self._selected_brief(),
                selected_id=self._selected_id,
                ready=ready,
            )
            try:
                self._probe_requests.put_nowait(request)
            except queue.Full:
                try:
                    self._probe_requests.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._probe_requests.put_nowait(request)
                except queue.Full:
                    pass
            self._last_probe_at = now
        except Exception as exc:
            print(f"[probe] 提交失败：{exc}", flush=True)

    def _disable_trt(self, exc: Exception) -> None:
        self._shutdown_probe_worker()
        name = self._engine_name or "TensorRT"
        self.trt = None
        self._engine_name = None
        self._runtime_label = "PyTorch"
        self._last_runtime = "pytorch"
        self._state_dirty = True
        message = f"{name} 运行失败，已回退 PyTorch：{exc}"
        self._last_error = message
        self.emit({"type": "error", "message": message})

    # ---------- lifecycle ----------

    def start(self) -> None:
        if self.thread and self.thread.is_alive():
            return
        self._stop.clear()
        self.thread = threading.Thread(target=self._run, name="yolo-engine", daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=3)
        self._shutdown_probe_worker()
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ---------- public control ----------

    def submit(self, command: dict[str, Any]) -> None:
        self._commands.put(command)

    def set_source(self, kind: str, value: str, label: str | None = None) -> None:
        self.submit({"type": "source", "kind": kind, "value": value, "label": label})

    def set_paused(self, paused: bool) -> None:
        self.submit({"type": "pause", "value": bool(paused)})

    def step(self) -> None:
        self.submit({"type": "step"})

    def request_hold(self) -> dict[str, Any]:
        with self._probe_lock:
            has_best = self._best is not None
        if not has_best:
            return {"ok": False, "error": "还没有捕捉到带人脸的帧，稍等再试"}
        self.submit({"type": "hold"})
        return {"ok": True, "best": self._best_info()}

    def release_hold(self) -> None:
        self.submit({"type": "release"})

    def select(self, track_id: int | None) -> None:
        self._selected_id = track_id
        self._state_dirty = True

    def enroll(self, name: str) -> dict[str, Any]:
        with self._crop_lock:
            crop = None if self._selected_aligned is None else self._selected_aligned.copy()
            owner = self._selected_owner
        if crop is None:
            return {"ok": False, "error": "还没有稳定的人脸可注册，请让人脸正对画面停留片刻"}
        if owner is None or owner != self._selected_id:
            return {"ok": False, "error": "选中的人脸正在切换，请稍候再试"}
        clean = (name or "").strip()[:24]
        if not clean:
            return {"ok": False, "error": "请输入姓名"}
        try:
            if self.gallery.items:
                known = {item["name"] for item in self.gallery.items}
                if clean in known:
                    clean = f"{clean} 2"
            item = self.gallery.add_from_aligned(clean, crop)
        except Exception as exc:
            return {"ok": False, "error": f"注册失败：{exc}"}
        self.submit({"type": "reidentify"})
        self._state_dirty = True
        self.emit({"type": "toast", "message": f"已注册「{clean}」"})
        return {"ok": True, "item": item}

    def remove_gallery(self, item_id: str) -> bool:
        ok = self.gallery.remove(item_id)
        if ok:
            self.submit({"type": "reidentify"})
            self._state_dirty = True
        return ok

    def snapshot(self) -> dict[str, Any]:
        return {
            "type": "hello",
            "state": self.state_payload(),
            "model": {**self.model_info, "meta": self.layer_meta},
            "live": self.latest_live,
            "probe": self.latest_probe,
        }

    def state_payload(self) -> dict[str, Any]:
        return {
            "paused": self._paused,
            "source": self.source.public(),
            "selected": self._selected_id,
            "gallery": self.gallery.public(),
            "stats": self.stats_payload(),
            "hold": self._hold_info(),
            "error": self._last_error,
        }

    def _best_info(self) -> dict[str, Any] | None:
        with self._probe_lock:
            best = self._best
        if best is None:
            return None
        return {
            "seq": best["seq"],
            "face_conf": best["face_conf"],
            "face_id": best["face_id"],
            "t": best["t"],
        }

    def _hold_info(self) -> dict[str, Any] | None:
        with self._probe_lock:
            held = self._held
        if held is None:
            return None
        return {**held, "active": True}

    def stats_payload(self) -> dict[str, Any]:
        now = time.monotonic()
        recent = [t for t in self._fps_marks if now - t < 2.5]
        fps = 0.0
        if len(recent) > 1:
            span = max(0.001, recent[-1] - recent[0])
            fps = (len(recent) - 1) / span
        memory = 0.0
        if self.device.type == "cuda":
            memory = torch.cuda.memory_allocated() / (1024 ** 2)
        return {
            "fps": round(fps, 1),
            "infer_ms": round(self._infer_ms, 1),
            "probe_ms": round(self._probe_ms, 1),
            "faces": self._visible_faces,
            "frame": self._frame_no,
            "gpu_mem_mb": round(memory, 0),
            "probe_age": round(now - self._last_probe_at, 1) if self._last_probe_at else None,
            "best": self._best_info(),
            "held": self._held is not None,
            "runtime": self._last_runtime,
        }

    # ---------- worker ----------

    def _run(self) -> None:
        self._open_source(self.source)
        last_status = 0.0
        while not self._stop.is_set():
            self._drain_commands()
            if self._state_dirty:
                self._emit_state()
            if self._paused and not self._step_once:
                self._stop.wait(0.06)
                if time.monotonic() - last_status > 1.0:
                    self.emit({"type": "status", **self.stats_payload()})
                    last_status = time.monotonic()
                continue
            if self._cap is None or not self._cap.isOpened():
                self._stop.wait(0.25)
                continue

            if self.source.end:
                position = float(self._cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
                if position >= self.source.end:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, self.source.start)

            ok, frame = self._cap.read()
            if not ok:
                if self.source.loop:
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                if self._last_error != "视频已播放结束":
                    self._last_error = "视频已播放结束"
                    self.emit({"type": "error", "message": "视频已播放结束，可切换来源继续"})
                    self._emit_state()
                    self._paused = True
                self._stop.wait(0.2)
                continue

            self._step_once = False
            try:
                self._process(frame)
            except Exception as exc:
                self._last_error = f"推理异常：{exc}"
                self.emit({"type": "error", "message": self._last_error})
                self._stop.wait(0.3)
                continue

            now = time.monotonic()
            if time.monotonic() - last_status > 0.9:
                self.emit({"type": "status", **self.stats_payload()})
                last_status = time.monotonic()

            if self._src_fps > 0:
                delay = self._next_due - now
                if delay > 0:
                    self._stop.wait(min(delay, 0.25))
                self._next_due += 1.0 / self._src_fps
                if self._next_due < now - 0.4:
                    self._next_due = now
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def _drain_commands(self) -> None:
        while True:
            try:
                cmd = self._commands.get_nowait()
            except queue.Empty:
                return
            kind = cmd.get("type")
            if kind == "source":
                self._switch_source(cmd)
            elif kind == "pause":
                self._paused = bool(cmd.get("value"))
                if not self._paused:
                    self._held = None
                self._state_dirty = True
            elif kind == "step":
                self._held = None
                self._paused = True
                self._step_once = True
                self._state_dirty = True
            elif kind == "hold":
                self._hold_best()
            elif kind == "release":
                if self._held is not None:
                    self._held = None
                    self._paused = False
                    self._state_dirty = True
            elif kind == "reidentify":
                for track in self.tracker.tracks.values():
                    track.identity = None
                    track.embed_age = 999
                self._state_dirty = True

    def _switch_source(self, cmd: dict[str, Any]) -> None:
        self._probe_generation += 1
        while True:
            try:
                self._probe_requests.get_nowait()
            except queue.Empty:
                break
        kind = str(cmd.get("kind") or "demo")
        value = str(cmd.get("value") or "")
        label = cmd.get("label") or None
        loop = kind in ("demo", "file")
        start = 0
        end = 0
        if kind == "demo":
            key = value if value in DEMO_SCENES else "closeup"
            scene = DEMO_SCENES[key]
            value = key
            label = label or scene["label"]
            start = int(scene["start"])
            end = int(scene["end"])
        elif kind == "file":
            label = label or Path(value).name
        elif kind == "webcam":
            label = label or f"摄像头 {value or 0}"
        elif kind == "stream":
            label = label or "网络视频流"
        spec = SourceSpec(
            kind=kind,
            value=value,
            label=label or kind,
            loop=loop,
            start=start,
            end=end,
        )
        cap = self._open_capture(spec)
        if cap is None or not cap.isOpened():
            self._last_error = f"无法打开来源：{spec.label}"
            self.emit({"type": "error", "message": self._last_error})
            self._emit_state()
            return
        if spec.start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, spec.start)
        if self._cap is not None:
            self._cap.release()
        self._cap = cap
        self.source = spec
        # 摄像头/流的读取本身就是节奏；DSHOW 常把 FPS 报成 0，再回退 15 会把 30fps 砍半。
        self._src_fps = 0.0 if kind in ("webcam", "stream") else self._read_fps(cap)
        self._next_due = 0.0
        self._last_error = ""
        self.tracker.clear()
        self.latest_probe = None
        self.latest_live = None
        self._best = None
        self._held = None
        self._selected_id = None
        self._visible_faces = 0
        with self._crop_lock:
            self._selected_aligned = None
            self._selected_owner = None
        self._last_probe_at = 0.0
        self._paused = False
        self._frame_no = 0
        self._fps_marks.clear()
        self.emit({"type": "toast", "message": f"已切换来源：{spec.label}"})
        self._state_dirty = True

    def _consider_best(
        self,
        live: dict[str, Any],
        probe: dict[str, Any],
        selected_id: int | None = None,
    ) -> None:
        faces = probe.get("faces") or []
        if not faces:
            return
        wanted = self._selected_id if selected_id is None else selected_id
        # Keep the best tied to the selected face so the HUD, ROI and microscope agree.
        star = next((face for face in faces if face.get("id") == wanted), None)
        if star is None:
            return
        conf = float(star.get("face_conf") or 0.0)
        if conf <= 0.0:
            return
        with self._probe_lock:
            if self._best is not None and conf <= self._best["face_conf"] + 1e-4:
                return
            self._best = {
                "live": live,
                "probe": probe,
                "face_id": int(star["id"]),
                "face_conf": round(conf, 4),
                "seq": int(probe["seq"]),
                "t": float(probe["t"]),
            }

    def _hold_best(self) -> None:
        with self._probe_lock:
            best = self._best
        if best is None:
            self.emit({"type": "error", "message": "还没有捕捉到带人脸的帧，稍等再试"})
            return
        face_id = best["face_id"]
        face_conf = best["face_conf"]
        info = self._best_info()
        held_faces = [
            {**face, "selected": face.get("id") == face_id}
            for face in best["probe"].get("faces") or []
        ]
        star = next((face for face in held_faces if face.get("selected")), None)
        held_live = {
            **best["live"],
            "held": True,
            "hold": info,
            "faces": [
                {**face, "selected": face.get("id") == face_id}
                for face in best["live"].get("faces") or []
            ],
            "selected": face_id,
        }
        held_probe = {
            **best["probe"],
            "held": True,
            "hold": info,
            "faces": held_faces,
            "face": (
                {"id": face_id, "face_conf": face_conf, "identity": star.get("identity")}
                if star is not None
                else best["probe"].get("face")
            ),
        }
        with self._probe_lock:
            self._held = info
        self._paused = True
        self._step_once = False
        self._selected_id = face_id
        self.latest_live = held_live
        self.latest_probe = held_probe
        self._emit_state()
        self.emit(held_live)
        self.emit(held_probe)
        self.emit({"type": "toast", "message": f"已定格最佳帧 · 置信度 {face_conf * 100:.1f}%"})

    def _open_source(self, spec: SourceSpec) -> None:
        cap = self._open_capture(spec)
        if cap is None or not cap.isOpened():
            self._last_error = f"无法打开来源：{spec.label}"
            self.emit({"type": "error", "message": self._last_error})
            return
        if spec.start:
            cap.set(cv2.CAP_PROP_POS_FRAMES, spec.start)
        self._cap = cap
        self._src_fps = 0.0 if spec.kind in ("webcam", "stream") else self._read_fps(cap)

    @staticmethod
    def _open_capture(spec: SourceSpec) -> cv2.VideoCapture | None:
        try:
            if spec.kind == "demo":
                return cv2.VideoCapture(str(ROOT / "assets" / "sample_faces.mp4"))
            if spec.kind == "webcam":
                index = int(spec.value or 0)
                cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if hasattr(cv2, "CAP_DSHOW") else cv2.VideoCapture(index)
                if not cap.isOpened():
                    cap.release()
                    cap = cv2.VideoCapture(index)
                return cap
            return cv2.VideoCapture(spec.value)
        except Exception:
            return None

    @staticmethod
    def _read_fps(cap: cv2.VideoCapture) -> float:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if not 1.0 <= fps <= 240.0:
            fps = 15.0
        return fps

    # ---------- per-frame ----------

    def _process(self, frame: np.ndarray) -> None:
        t0 = time.perf_counter()
        height, width = frame.shape[:2]
        tensor = torch.from_numpy(frame).to(self.device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).contiguous().float().div_(255.0)
        tensor = tensor[:, [2, 1, 0]]
        tensor = F.interpolate(
            tensor, size=(IMG_SIZE, IMG_SIZE), mode="bilinear", align_corners=False
        )

        now = time.monotonic()
        selected_before = self.tracker.get(self._selected_id)
        roi = None
        if selected_before is not None:
            box = selected_before.bbox
            roi = (
                float(np.clip(box[0] / width, 0, 1)),
                float(np.clip(box[1] / height, 0, 1)),
                float(np.clip(box[2] / width, 0, 1)),
                float(np.clip(box[3] / height, 0, 1)),
            )
        probe_worker_ready = (
            self._probe_async
            and self.trt is not None
            and self._probe_thread is not None
            and self._probe_thread.is_alive()
        )
        probe_interval = PROBE_INTERVAL if probe_worker_ready else PROBE_FALLBACK_INTERVAL
        want_probe = (now - self._last_probe_at) >= probe_interval
        async_probe = want_probe and probe_worker_ready
        # 没有 TensorRT 时探针必须和实时帧共用同一次 PyTorch 前向，只能内联。
        probe_inline = want_probe and not async_probe
        probe_started = 0.0
        if probe_inline:
            self.taps.begin_probe(roi)
            probe_started = time.perf_counter()

        use_trt = self.trt is not None and not probe_inline
        try:
            if use_trt:
                raw = self.trt.run(tensor)
                self._last_runtime = "tensorrt"
            else:
                with torch.inference_mode():
                    output = self.torch_model(tensor)
                raw = output[0] if isinstance(output, (list, tuple)) else output
                self._last_runtime = "pytorch"
        except Exception as exc:
            if not use_trt:
                raise
            self._disable_trt(exc)
            probe_inline = want_probe
            if probe_inline:
                self.taps.begin_probe(roi)
                probe_started = time.perf_counter()
            with torch.inference_mode():
                output = self.torch_model(tensor)
            raw = output[0] if isinstance(output, (list, tuple)) else output
            self._last_runtime = "pytorch"
        raw_np = raw[0].float().cpu().numpy()
        self._infer_ms = (time.perf_counter() - t0) * 1000.0

        probe_payload = None
        if probe_inline:
            probe = self.taps.finish_probe()
            self._probe_ms = (time.perf_counter() - probe_started) * 1000.0
            self._last_probe_at = now
            probe_payload = {
                "type": "probe",
                "seq": self._frame_no,
                "t": round(time.time(), 3),
                "maps": probe["maps"],
                "micro": probe["micro"],
                "head": probe["head"],
                "roi": roi,
                "took_ms": round(self._probe_ms, 1),
            }

        detections = self._parse(raw_np, frame, width, height)
        tracks = self.tracker.update(detections, time.time())
        self._visible_faces = len(tracks)
        self._update_identity(frame, tracks)

        visible_ids = {track.id for track in tracks}
        if self._selected_id not in visible_ids:
            best = max(tracks, key=lambda track: track.face_conf) if tracks else None
            self._selected_id = best.id if best is not None else None

        selected = self.tracker.get(self._selected_id)
        if selected is not None and selected.last_aligned is not None:
            with self._crop_lock:
                self._selected_aligned = selected.last_aligned.copy()
                self._selected_owner = selected.id
        elif selected is None:
            with self._crop_lock:
                self._selected_aligned = None
                self._selected_owner = None

        self._frame_no += 1
        now_mono = time.monotonic()
        self._fps_marks.append(now_mono)
        if len(self._fps_marks) > 90:
            self._fps_marks = self._fps_marks[-90:]

        jpeg = self._encode_frame(frame)
        faces = [self._face_payload(track, width, height) for track in tracks]
        live = {
            "type": "live",
            "seq": self._frame_no,
            "t": round(time.time(), 3),
            "frame": jpeg,
            "faces": faces,
            "selected": self._selected_id,
            "source": self.source.public(),
            "stats": {
                "infer_ms": round(self._infer_ms, 1),
                "probe_ms": round(self._probe_ms, 1) if probe_inline else None,
                "fps": self.stats_payload()["fps"],
                "probed": want_probe,
                "runtime": self._last_runtime,
            },
        }
        self.latest_live = live
        self.emit(live)

        if probe_payload is not None:
            probe_payload.update(
                seq=self._frame_no,
                frame=jpeg,
                faces=faces,
                face=self._selected_brief(),
            )
            self._consider_best(live, probe_payload, self._selected_id)
            self.latest_probe = probe_payload
            self.emit(probe_payload)
        elif want_probe and async_probe:
            self._submit_probe(tensor, roi, live, now)

        if self._frame_no % 3 == 0:
            head_live = self.taps._head_payload(raw_np)
            if head_live is not None:
                self.emit(
                    {
                        "type": "head",
                        "seq": self._frame_no,
                        "t": round(time.time(), 3),
                        "head": head_live,
                        "selected": self._selected_id,
                    }
                )

    def _parse(self, raw: np.ndarray, frame: np.ndarray, width: int, height: int) -> list[dict[str, Any]]:
        if raw.ndim != 2 or raw.shape[1] < 57:
            return []
        scores = raw[:, 4]
        keep = np.where(scores >= PERSON_CONF)[0]
        if len(keep) == 0:
            return []
        keep = keep[np.argsort(scores[keep])[::-1][:12]]
        sx = width / float(IMG_SIZE)
        sy = height / float(IMG_SIZE)
        detections: list[dict[str, Any]] = []
        for index in keep:
            row = raw[index]
            box = row[:4].astype(np.float32).copy()
            box[[0, 2]] *= sx
            box[[1, 3]] *= sy
            box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
            if box[2] - box[0] < 8 or box[3] - box[1] < 8:
                continue
            keypoints = row[6:57].reshape(17, 3).astype(np.float32).copy()
            keypoints[:, 0] *= sx
            keypoints[:, 1] *= sy
            face = face_box_from_pose(keypoints, width, height)
            if face is None:
                continue
            face_box, kp_conf, eye_conf = face
            person_conf = float(row[4])
            detections.append(
                {
                    "bbox": face_box.astype(np.float32),
                    "person_box": box,
                    "keypoints": keypoints,
                    "person_conf": person_conf,
                    "kp_conf": kp_conf,
                    "eye_conf": eye_conf,
                    "face_conf": float(np.clip(person_conf * kp_conf, 0.0, 1.0)),
                }
            )
        return detections

    def _update_identity(self, frame: np.ndarray, tracks: list[FaceTrack]) -> None:
        budget = 2 if self._frame_no % 4 == 0 else 0
        for track in tracks:
            if budget <= 0:
                break
            if track.identity is not None and track.embed_age < 20:
                continue
            aligned = align_face_from_pose(frame, track.keypoints)
            if aligned is None:
                continue
            try:
                embedding = self.embedder.embed_aligned(aligned)
                match = self.gallery.match(embedding, MATCH_THRESHOLD)
            except Exception:
                continue
            budget -= 1
            similarity = float(match["similarity"]) if match else 0.0
            previous = track.identity
            if previous is not None and previous.get("similarity"):
                similarity = round(previous["similarity"] * 0.6 + similarity * 0.4, 4)
            else:
                similarity = round(similarity, 4)
            known = bool(match and match["known"] and similarity >= MATCH_THRESHOLD * 0.92)
            track.identity = {
                "known": known,
                "status": "known" if known else "unknown",
                "name": match["best_name"] if match else "未注册",
                "best_name": match["best_name"] if match else None,
                "similarity": similarity,
            }
            track.last_aligned = aligned
            track.embed_age = 0
        for track in tracks:
            track.embed_age += 1

    # ---------- payloads ----------

    def _face_payload(self, track: FaceTrack, width: int, height: int) -> dict[str, Any]:
        box = track.bbox
        identity = track.identity
        if identity is None:
            identity_payload = {"status": "pending", "known": False, "name": "识别中…", "similarity": None, "best_name": None}
        else:
            identity_payload = {
                "status": identity["status"],
                "known": bool(identity["known"]),
                "name": identity.get("name") or "未注册",
                "similarity": identity.get("similarity"),
                "best_name": identity.get("best_name"),
            }
        crop = ""
        if track.last_aligned is not None:
            ok, buffer = cv2.imencode(".jpg", track.last_aligned, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if ok:
                crop = base64.b64encode(buffer.tobytes()).decode("ascii")
        return {
            "id": track.id,
            "box": [
                round(float(box[0] / width), 5),
                round(float(box[1] / height), 5),
                round(float(box[2] / width), 5),
                round(float(box[3] / height), 5),
            ],
            "keypoints": [
                [
                    round(float(point[0] / width), 5),
                    round(float(point[1] / height), 5),
                    round(float(point[2]), 4),
                ]
                for point in track.keypoints
            ],
            "person_conf": round(float(track.person_conf), 4),
            "face_conf": round(float(track.face_conf), 4),
            "kp_conf": round(float(track.kp_conf), 4),
            "eye_conf": round(float(track.eye_conf), 4),
            "identity": identity_payload,
            "crop": crop,
            "selected": track.id == self._selected_id,
        }

    def _selected_brief(self) -> dict[str, Any] | None:
        track = self.tracker.get(self._selected_id)
        if track is None:
            return None
        return {
            "id": track.id,
            "face_conf": round(float(track.face_conf), 4),
            "identity": track.identity or {"status": "pending", "name": "识别中…"},
        }

    def _encode_frame(self, frame: np.ndarray) -> dict[str, Any]:
        height, width = frame.shape[:2]
        # 竖屏视频只按宽度缩会得到 960×1707，JPEG 和传输成本都会翻倍。
        scale = min(1.0, DISPLAY_WIDTH / width, DISPLAY_HEIGHT / height)
        display = frame
        if scale < 1.0:
            display = cv2.resize(
                frame,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        ok, buffer = cv2.imencode(".jpg", display, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        encoded = base64.b64encode(buffer.tobytes()).decode("ascii") if ok else ""
        out_h, out_w = display.shape[:2]
        return {"w": int(out_w), "h": int(out_h), "jpeg": encoded}

    def _emit_state(self) -> None:
        self._state_dirty = False
        self.emit({"type": "state", **self.state_payload()})
