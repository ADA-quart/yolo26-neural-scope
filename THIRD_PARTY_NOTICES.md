# Third-party notices and license map

This repository combines source code, model weights and demo media that are
licensed differently. The table below is the short version; the sections after
it explain the practical consequences.

| Component | Where | License |
|---|---|---|
| This repository's source code | `backend/`, `frontend/src/`, `run.ps1` | **AGPL-3.0** (see `LICENSE`) |
| YOLO26n-Pose weights + Ultralytics runtime | `weights/yolo26n-pose.pt`, `ultralytics` | **AGPL-3.0** |
| EdgeFace-S ONNX weights | `weights/edgeface_s_gamma_05.onnx` | **CC BY-NC-SA 4.0** |
| EdgeFace inference code (upstream) | not bundled | BSD-3-Clause |
| Demo video + derived face crops | `assets/` | **CC BY 4.0** |
| PyTorch / torchvision | dependency | BSD-3-Clause |
| ONNX Runtime | dependency | MIT |
| OpenCV (`opencv-python`) | dependency | Apache-2.0 |
| NumPy | dependency | BSD-3-Clause |
| FastAPI | dependency | MIT |
| Starlette | dependency | BSD-3-Clause |
| Uvicorn / websockets | dependency | BSD-3-Clause |
| Pydantic | dependency | MIT |
| python-multipart | dependency | Apache-2.0 |
| React / React DOM | dependency | MIT |
| Vite | dependency | MIT |
| TypeScript | dependency | Apache-2.0 |
| framer-motion | dependency | MIT |
| lucide-react | dependency | ISC |

No fonts are bundled; the UI uses system fonts (`Segoe UI`, `Microsoft YaHei
UI`, `PingFang SC`, `Bahnschrift`, `Cascadia Code`, `Consolas`).

## 1. Code: AGPL-3.0

The backend imports Ultralytics and the repository distributes YOLO26 weights,
so the project is released under the GNU Affero General Public License v3.0.
In plain language:

- You may use, study, modify and redistribute the code.
- If you redistribute it (or a modified version), you must keep it under
  AGPL-3.0 and provide the corresponding source.
- **AGPL §13 (network use):** if you run a modified version as a network
  service, users interacting with it over the network must be able to obtain
  the corresponding source code.
- There is no warranty.

This is a summary, not legal advice. The binding terms are in `LICENSE`.

## 2. YOLO26 / Ultralytics: AGPL-3.0

- Upstream: <https://github.com/ultralytics/ultralytics>
- Model docs: <https://docs.ultralytics.com/models/yolo26>
- Weights source:
  <https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-pose.pt>
- License: GNU AGPL-3.0.

Commercial or closed-source deployment requires an Ultralytics Enterprise
License, or a different detector whose license permits your use case.

## 3. EdgeFace: BSD-3 code, CC BY-NC-SA 4.0 weights

- Paper: *EdgeFace: Efficient Face Recognition Model for Edge Devices*,
  <https://arxiv.org/abs/2307.01838>
- Inference code: <https://github.com/otroshi/edgeface> — BSD-3-Clause.
- Pretrained model card:
  <https://huggingface.co/Idiap/EdgeFace-XS-GAMMA> — **CC BY-NC-SA 4.0**.
- Bundled file: `weights/edgeface_s_gamma_05.onnx`, a format conversion of the
  EdgeFace-S (γ = 0.5) checkpoint.

The **NC** clause means the bundled face-recognition weights may not be used
for commercial advantage or monetary compensation. The **SA** clause requires
shared adaptations of the weights to keep the same license. If you need
commercial use, replace this embedder with a model whose weights allow it, or
obtain a separate license from Idiap.

## 4. Demo media: CC BY 4.0

- Source: <https://github.com/intel-iot-devkit/sample-videos>
- File: `face-demographics-walking-and-pause.mp4`
- License: <https://creativecommons.org/licenses/by/4.0/>
- Attribution: Intel IoT DevKit sample-videos.
- Changes: face crops in `assets/demo_faces/` were extracted and resized from
  the video; the app plays two trimmed frame ranges as demo scenes.

See `assets/README.md` for details. The demo is included for teaching and
testing only; do not use it for surveillance or identity profiling.

## 5. If you redistribute or deploy this project

1. Keep `LICENSE`, `NOTICE`, `THIRD_PARTY_NOTICES.md`, `assets/README.md`,
   `weights/README.md` and the CC BY attribution intact.
2. Make the source available to network users if you deploy a modified version
   (AGPL §13).
3. Do not describe the bundled EdgeFace weights as commercial-use or
   open-source-approved; they are non-commercial.
4. Do not use the demo footage to imply endorsement by the people in it.
5. For commercial use, obtain an Ultralytics commercial license and replace the
   EdgeFace weights (or obtain permission).
