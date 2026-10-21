# Model weights

These files are tracked on purpose so the teaching demo runs without a
download step. They are third-party artifacts and are **not** covered by the
repository's AGPL-3.0 code license.

| File | Model | Upstream license | Notes |
|---|---|---|---|
| `yolo26n-pose.pt` (7.6 MB) | YOLO26n-Pose | **AGPL-3.0** | Ultralytics asset release `v8.4.0` |
| `edgeface_s_gamma_05.onnx` (15 MB) | EdgeFace-S, γ = 0.5 | **CC BY-NC-SA 4.0** | Idiap Research Institute; non-commercial |

## YOLO26n-Pose

- Project: <https://github.com/ultralytics/ultralytics>
- Docs: <https://docs.ultralytics.com/models/yolo26>
- Download: <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt>
- License: GNU AGPL-3.0. Commercial or closed-source use requires an
  Ultralytics Enterprise License.

## EdgeFace-S

- Paper: *EdgeFace: Efficient Face Recognition Model for Edge Devices*
  (<https://arxiv.org/abs/2307.01838>)
- Code: <https://github.com/otroshi/edgeface> (BSD-3-Clause)
- Pretrained weights: <https://huggingface.co/Idiap/EdgeFace-XS-GAMMA>
  — **CC BY-NC-SA 4.0**
- The ONNX file is a format conversion of the EdgeFace-S (γ = 0.5) checkpoint
  used by this demo. It must not be used for commercial purposes.

If you need a commercially usable version, replace EdgeFace with an embedder
whose weights permit commercial use, or obtain the appropriate licenses from
the rights holders. See `../THIRD_PARTY_NOTICES.md` for details.
