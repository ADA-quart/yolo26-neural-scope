from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@dataclass
class FaceTrack:
    id: int
    bbox: np.ndarray
    keypoints: np.ndarray
    person_conf: float
    face_conf: float
    kp_conf: float
    eye_conf: float
    first_seen: float
    last_seen: float
    hits: int = 1
    last_aligned: np.ndarray | None = None
    identity: dict[str, Any] | None = None
    embed_age: int = 999
    extra: dict[str, Any] = field(default_factory=dict)


class FaceTracker:
    def __init__(self, iou_threshold: float = 0.22, max_age: float = 1.6, publish_age: float = 0.22):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.publish_age = publish_age
        self.tracks: dict[int, FaceTrack] = {}
        self.next_id = 1

    def update(self, detections: list[dict[str, Any]], now: float) -> list[FaceTrack]:
        unmatched = set(range(len(detections)))
        matches: dict[int, int] = {}

        for track_id, track in sorted(self.tracks.items(), key=lambda x: x[1].bbox[0]):
            best_index = None
            best_iou = self.iou_threshold
            for index in unmatched:
                value = _iou(track.bbox, detections[index]["bbox"])
                if value > best_iou:
                    best_iou = value
                    best_index = index
            if best_index is not None:
                matches[track_id] = best_index
                unmatched.discard(best_index)

        for track_id, index in matches.items():
            track = self.tracks[track_id]
            detection = detections[index]
            old_box = track.bbox
            new_box = detection["bbox"]
            track.bbox = old_box * 0.55 + new_box * 0.45
            track.keypoints = detection["keypoints"]
            track.person_conf = track.person_conf * 0.7 + detection["person_conf"] * 0.3
            track.face_conf = track.face_conf * 0.65 + detection["face_conf"] * 0.35
            track.kp_conf = track.kp_conf * 0.7 + detection["kp_conf"] * 0.3
            track.eye_conf = detection["eye_conf"]
            track.last_seen = now
            track.hits += 1
            track.embed_age += 1

        for index in unmatched:
            detection = detections[index]
            track = FaceTrack(
                id=self.next_id,
                bbox=detection["bbox"].copy(),
                keypoints=detection["keypoints"].copy(),
                person_conf=detection["person_conf"],
                face_conf=detection["face_conf"],
                kp_conf=detection["kp_conf"],
                eye_conf=detection["eye_conf"],
                first_seen=now,
                last_seen=now,
            )
            self.tracks[track.id] = track
            self.next_id += 1

        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if now - track.last_seen > self.max_age
        ]
        for track_id in expired:
            self.tracks.pop(track_id, None)

        return sorted(
            (track for track in self.tracks.values() if now - track.last_seen <= self.publish_age),
            key=lambda track: track.id,
        )

    def get(self, track_id: int | None) -> FaceTrack | None:
        if track_id is None:
            return None
        return self.tracks.get(track_id)

    def clear(self) -> None:
        self.tracks.clear()
        self.next_id = 1
