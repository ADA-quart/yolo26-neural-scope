from __future__ import annotations

import asyncio
import contextlib
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from engine import Engine

ROOT = Path(__file__).resolve().parents[1]
UPLOAD_DIR = ROOT / "state" / "uploads"
FRONTEND_DIST = ROOT / "frontend" / "dist"
ALLOWED_VIDEO = {".mp4", ".webm", ".mov", ".avi", ".mkv", ".m4v"}


class Hub:
    def __init__(self) -> None:
        self.queues: set[asyncio.Queue[dict[str, Any]]] = set()
        self.loop: asyncio.AbstractEventLoop | None = None

    def bind(self) -> None:
        self.loop = asyncio.get_running_loop()

    def register(self) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=14)
        self.queues.add(queue)
        return queue

    def unregister(self, queue: asyncio.Queue[dict[str, Any]]) -> None:
        self.queues.discard(queue)

    def broadcast_threadsafe(self, message: dict[str, Any]) -> None:
        if self.loop is None:
            return
        self.loop.call_soon_threadsafe(self._broadcast, message)

    def _broadcast(self, message: dict[str, Any]) -> None:
        for queue in list(self.queues):
            if queue.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    queue.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(message)


hub = Hub()
engine: Engine | None = None


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    hub.bind()
    engine = Engine(emit=hub.broadcast_threadsafe)
    engine.start()
    try:
        yield
    finally:
        engine.stop()
        engine = None


app = FastAPI(title="YOLO26 神经显微镜", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:8765"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_engine() -> Engine:
    if engine is None:
        raise HTTPException(503, "引擎尚未就绪")
    return engine


class SourceBody(BaseModel):
    kind: str
    value: str | None = None


class ControlBody(BaseModel):
    action: str


class SelectBody(BaseModel):
    id: int | None = None


class EnrollBody(BaseModel):
    name: str


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    eng = get_engine()
    return {
        "ok": True,
        "state": eng.state_payload(),
        "model": {**eng.model_info, "meta": eng.layer_meta},
        "probe": eng.latest_probe,
    }


@app.get("/api/gallery")
def api_gallery() -> dict[str, Any]:
    eng = get_engine()
    return {"items": eng.gallery.public()}


@app.post("/api/source")
def api_source(body: SourceBody) -> dict[str, Any]:
    eng = get_engine()
    kind = body.kind
    if kind not in {"demo", "file", "webcam", "stream"}:
        raise HTTPException(400, "不支持的来源类型")
    value = (body.value or "").strip()
    if kind == "demo":
        eng.set_source("demo", value or "closeup")
        return {"ok": True}
    if kind == "webcam":
        eng.set_source("webcam", value or "0")
        return {"ok": True}
    if not value:
        raise HTTPException(400, "缺少来源地址")
    if kind == "file":
        path = Path(value)
        if not path.is_absolute():
            path = (ROOT / value).resolve()
        if not path.exists() or path.suffix.lower() not in ALLOWED_VIDEO:
            raise HTTPException(400, "视频文件不存在或格式不支持")
        eng.set_source("file", str(path), path.name)
        return {"ok": True}
    if not value.lower().startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://")):
        raise HTTPException(400, "视频流地址需要以 rtsp/rtmp/http(s)/udp 开头")
    eng.set_source("stream", value)
    return {"ok": True}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)) -> dict[str, Any]:
    eng = get_engine()
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_VIDEO:
        raise HTTPException(400, "只支持 mp4 / webm / mov / avi / mkv 视频")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    target = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{suffix}"
    total = 0
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                total += len(chunk)
                if total > 500 * 1024 * 1024:
                    raise HTTPException(413, "视频超过 500MB 限制")
                handle.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    label = Path(file.filename or target.name).stem[:40]
    eng.set_source("file", str(target), label)
    return {"ok": True, "path": str(target.relative_to(ROOT)), "bytes": total, "label": label}


@app.post("/api/control")
def api_control(body: ControlBody) -> dict[str, Any]:
    eng = get_engine()
    if body.action == "pause":
        eng.set_paused(True)
    elif body.action == "resume":
        eng.set_paused(False)
    elif body.action == "step":
        eng.step()
    elif body.action == "hold":
        result = eng.request_hold()
        if not result.get("ok"):
            raise HTTPException(400, str(result.get("error") or "还没有可定格的帧"))
        return {"ok": True, "action": "hold", "best": result.get("best")}
    elif body.action == "release":
        eng.release_hold()
    else:
        raise HTTPException(400, "未知操作")
    return {"ok": True, "action": body.action}


@app.post("/api/select")
def api_select(body: SelectBody) -> dict[str, Any]:
    eng = get_engine()
    eng.select(body.id)
    return {"ok": True, "selected": body.id}


@app.post("/api/enroll")
def api_enroll(body: EnrollBody) -> dict[str, Any]:
    eng = get_engine()
    result = eng.enroll(body.name)
    if not result.get("ok"):
        raise HTTPException(400, str(result.get("error") or "注册失败"))
    return result


@app.delete("/api/gallery/{item_id}")
def api_gallery_delete(item_id: str) -> dict[str, Any]:
    eng = get_engine()
    if not eng.remove_gallery(item_id):
        raise HTTPException(404, "不存在的人脸档案")
    return {"ok": True}


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    return {"ok": engine is not None}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    global engine
    if engine is None:
        await websocket.close(code=1013)
        return
    queue = hub.register()
    try:
        with contextlib.suppress(Exception):
            queue.put_nowait(engine.snapshot())

        async def sender() -> None:
            while True:
                message = await queue.get()
                await websocket.send_json(message)

        async def receiver() -> None:
            while True:
                await websocket.receive_text()

        send_task = asyncio.create_task(sender())
        recv_task = asyncio.create_task(receiver())
        done, pending = await asyncio.wait({send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        for task in done:
            with contextlib.suppress(Exception):
                task.result()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        hub.unregister(queue)


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="frontend-assets")

    def index_response() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html", headers={"Cache-Control": "no-cache"})

    @app.get("/live")
    def live_page() -> FileResponse:
        return index_response()

    @app.get("/neural")
    def neural_page() -> FileResponse:
        return index_response()

    @app.get("/")
    def index_page() -> FileResponse:
        return index_response()


state_dir = ROOT / "state"
state_dir.mkdir(parents=True, exist_ok=True)
app.mount("/state", StaticFiles(directory=state_dir), name="state")


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse({"ok": False, "error": exc.detail}, status_code=exc.status_code)
