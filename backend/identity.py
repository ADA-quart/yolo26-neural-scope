from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime as ort


REFERENCE_5P = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


class FaceEmbedder:
    """EdgeFace-S ONNX embedding, CPU is enough for 1-2 faces."""

    def __init__(self, model_path: str | Path, threads: int = 2):
        options = ort.SessionOptions()
        options.intra_op_num_threads = max(1, threads)
        options.inter_op_num_threads = 1
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def embed_aligned(self, aligned_bgr: np.ndarray) -> np.ndarray:
        if aligned_bgr.shape[0] != 112 or aligned_bgr.shape[1] != 112:
            aligned_bgr = cv2.resize(aligned_bgr, (112, 112), interpolation=cv2.INTER_LINEAR)
        blob = cv2.dnn.blobFromImage(
            aligned_bgr,
            scalefactor=1.0 / 127.5,
            size=(112, 112),
            mean=(127.5, 127.5, 127.5),
            swapRB=True,
        )
        vec = self.session.run([self.output_name], {self.input_name: blob})[0]
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        return vec / max(float(np.linalg.norm(vec)), 1e-8)


def align_face_from_pose(image: np.ndarray, keypoints: np.ndarray) -> np.ndarray | None:
    """COCO pose -> ArcFace-style 112x112 crop using eyes + nose.

    COCO left/right are person-centric, so the image-left eye is keypoint 2.
    """
    if keypoints is None or len(keypoints) < 5:
        return None
    right_eye = keypoints[2]
    left_eye = keypoints[1]
    nose = keypoints[0]
    if min(float(right_eye[2]), float(left_eye[2]), float(nose[2])) < 0.12:
        return None
    eye_dist = float(np.linalg.norm(right_eye[:2] - left_eye[:2]))
    if eye_dist < 6.0:
        return None
    src = np.stack([right_eye[:2], left_eye[:2], nose[:2]]).astype(np.float32)
    matrix, _ = cv2.estimateAffinePartial2D(
        src, REFERENCE_5P[[0, 1, 2]], method=cv2.LMEDS
    )
    if matrix is None:
        return None
    scale = float(np.hypot(matrix[0, 0], matrix[0, 1]))
    if not 0.02 < scale < 30.0:
        return None
    return cv2.warpAffine(
        image,
        matrix,
        (112, 112),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def face_box_from_pose(
    keypoints: np.ndarray, frame_w: int, frame_h: int
) -> tuple[np.ndarray, float, float] | None:
    """Derive a face box + face confidence from COCO pose keypoints."""
    if keypoints is None or len(keypoints) < 5:
        return None
    face_points = keypoints[:5]
    nose, left_eye, right_eye = face_points[0], face_points[1], face_points[2]
    eye_conf = min(float(left_eye[2]), float(right_eye[2]))
    kp_conf = float(np.mean(np.clip(face_points[:, 2], 0.0, 1.0)))
    if eye_conf > 0.12:
        eye_dist = float(np.linalg.norm(left_eye[:2] - right_eye[:2]))
        if eye_dist >= 4.0:
            cx = float((left_eye[0] + right_eye[0]) * 0.5)
            eye_y = float((left_eye[1] + right_eye[1]) * 0.5)
            width = float(np.clip(eye_dist * 2.65, 22.0, min(frame_w, frame_h) * 0.85))
            height = width * 1.18
            cy = eye_y + height * 0.14
            box = np.array(
                [
                    cx - width * 0.5,
                    cy - height * 0.5,
                    cx + width * 0.5,
                    cy + height * 0.5,
                ],
                dtype=np.float32,
            )
            box[[0, 2]] = np.clip(box[[0, 2]], 0, frame_w - 1)
            box[[1, 3]] = np.clip(box[[1, 3]], 0, frame_h - 1)
            return box, kp_conf, eye_conf

    valid = face_points[face_points[:, 2] > 0.08]
    if len(valid) < 2:
        return None
    x1, y1 = np.min(valid[:, :2], axis=0)
    x2, y2 = np.max(valid[:, :2], axis=0)
    pad_x = max(12.0, (x2 - x1) * 0.65)
    pad_y = max(14.0, (y2 - y1) * 0.75)
    box = np.array([x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y], dtype=np.float32)
    box[[0, 2]] = np.clip(box[[0, 2]], 0, frame_w - 1)
    box[[1, 3]] = np.clip(box[[1, 3]], 0, frame_h - 1)
    if box[2] - box[0] < 10 or box[3] - box[1] < 10:
        return None
    return box, kp_conf, eye_conf


class FaceGallery:
    def __init__(self, state_dir: str | Path, embedder: FaceEmbedder):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.thumb_dir = self.state_dir / "faces"
        self.thumb_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.state_dir / "gallery.json"
        self.embedder = embedder
        self.lock = threading.RLock()
        self.items: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            self.items = [x for x in data.get("items", []) if x.get("embedding")]
        except Exception:
            self.items = []

    def _save(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"items": self.items}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        tmp.replace(self.path)

    def ensure_demo(self, image_path: str | Path, name: str = "示例人物 A") -> None:
        with self.lock:
            if self.items:
                return
            image = cv2.imread(str(image_path))
            if image is None:
                return
            try:
                embedding = self.embedder.embed_aligned(image)
            except Exception:
                return
            self._add_locked(name, image, embedding, demo=True)

    def ensure_demo_many(self, entries: list[tuple[str | Path, str]]) -> None:
        with self.lock:
            if self.items:
                return
            for image_path, name in entries:
                image = cv2.imread(str(image_path))
                if image is None:
                    continue
                try:
                    embedding = self.embedder.embed_aligned(image)
                except Exception:
                    continue
                self._add_locked(name, image, embedding, demo=True)

    def add_from_aligned(self, name: str, aligned_bgr: np.ndarray) -> dict[str, Any]:
        embedding = self.embedder.embed_aligned(aligned_bgr)
        with self.lock:
            return self._add_locked(name, aligned_bgr, embedding, demo=False)

    def _add_locked(
        self,
        name: str,
        aligned_bgr: np.ndarray,
        embedding: np.ndarray,
        demo: bool,
    ) -> dict[str, Any]:
        item_id = uuid.uuid4().hex[:10]
        thumb_name = f"{item_id}.jpg"
        cv2.imwrite(str(self.thumb_dir / thumb_name), aligned_bgr)
        item = {
            "id": item_id,
            "name": (name or "未命名").strip()[:24] or "未命名",
            "embedding": [float(x) for x in embedding],
            "thumb": thumb_name,
            "demo": bool(demo),
            "created": int(__import__("time").time()),
        }
        self.items.append(item)
        self._save()
        return self._public_item(item)

    @staticmethod
    def _public_item(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": item["id"],
            "name": item["name"],
            "demo": bool(item.get("demo")),
            "created": item.get("created"),
            "thumb": f"/state/faces/{item['thumb']}",
        }

    def public(self) -> list[dict[str, Any]]:
        with self.lock:
            return [self._public_item(item) for item in self.items]

    def match(self, embedding: np.ndarray, threshold: float) -> dict[str, Any] | None:
        with self.lock:
            if not self.items:
                return None
            matrix = np.asarray([item["embedding"] for item in self.items], dtype=np.float32)
            similarities = matrix @ embedding
            index = int(np.argmax(similarities))
            similarity = float(similarities[index])
            item = self.items[index]
            return {
                "name": item["name"] if similarity >= threshold else None,
                "best_name": item["name"],
                "similarity": similarity,
                "known": similarity >= threshold,
            }

    def remove(self, item_id: str) -> bool:
        with self.lock:
            for index, item in enumerate(self.items):
                if item["id"] == item_id:
                    self.items.pop(index)
                    try:
                        (self.thumb_dir / item["thumb"]).unlink(missing_ok=True)
                    except Exception:
                        pass
                    self._save()
                    return True
        return False
