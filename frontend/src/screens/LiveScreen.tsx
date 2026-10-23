import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  Camera,
  Check,
  ChevronDown,
  Clock3,
  Cpu,
  Expand,
  Link2,
  Pause,
  Play,
  Radar,
  ScanFace,
  SkipForward,
  Sparkles,
  Trash2,
  Trophy,
  UploadCloud,
  UserPlus,
  Video,
  Wifi,
  WifiOff,
  X,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Brand, ConnectionPill } from "../components/Brand";
import { Gauge } from "../components/Gauge";
import { Sparkline } from "../components/Sparkline";
import { ThemeToggle } from "../components/ThemeToggle";
import { Toasts } from "../components/Toasts";
import { useFeed } from "../hooks/useFeed";
import { api, num, pct, simLabel } from "../lib/utils";
import type { FaceInfo, GalleryItem } from "../types";

const FACE_KEYS = [0, 1, 2, 3, 4];

export default function LiveScreen() {
  const feed = useFeed();
  const { live, state, model, stats, connected, toasts, notify, dismiss } = feed;
  const [history, setHistory] = useState<number[]>([]);
  const [historyFor, setHistoryFor] = useState<number | null>(null);
  const [menuOpen, setMenuOpen] = useState(false);
  const [streamUrl, setStreamUrl] = useState("");
  const [busy, setBusy] = useState(false);
  const [enrollOpen, setEnrollOpen] = useState(false);
  const [name, setName] = useState("");
  const fileRef = useRef<HTMLInputElement>(null);
  const facesRef = useRef<FaceInfo[]>([]);
  const pausedRef = useRef(false);
  const holdRef = useRef(false);

  const faces = live?.faces ?? [];
  const selected =
    faces.find((face) => face.selected) ??
    faces.find((face) => face.id === (live?.selected ?? state?.selected)) ??
    faces[0] ??
    null;
  const paused = state?.paused ?? false;
  const hold = state?.hold ?? null;
  const best = stats?.best ?? null;
  const gallery = state?.gallery ?? [];
  const source = live?.source ?? state?.source;
  const frame = live?.frame;
  facesRef.current = faces;
  pausedRef.current = paused;
  holdRef.current = !!hold;

  useEffect(() => {
    if (selected && selected.id !== historyFor) {
      setHistoryFor(selected.id);
      setHistory([]);
    }
  }, [selected, historyFor]);

  useEffect(() => {
    if (!live || !selected) return;
    setHistory((values) => [...values.slice(-71), selected.face_conf]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live?.seq, selected?.id]);

  const selectFace = useCallback(
    async (id: number) => {
      try {
        await api("/api/select", { method: "POST", body: JSON.stringify({ id }) });
      } catch (error) {
        notify((error as Error).message, "error");
      }
    },
    [notify],
  );

  const control = useCallback(
    async (action: "pause" | "resume" | "step" | "hold" | "release") => {
      try {
        await api("/api/control", { method: "POST", body: JSON.stringify({ action }) });
      } catch (error) {
        notify((error as Error).message, "error");
      }
    },
    [notify],
  );

  const switchDemo = async (scene: string) => {
    setMenuOpen(false);
    try {
      await api("/api/source", { method: "POST", body: JSON.stringify({ kind: "demo", value: scene }) });
    } catch (error) {
      notify((error as Error).message, "error");
    }
  };

  const useWebcam = async () => {
    setMenuOpen(false);
    try {
      await api("/api/source", { method: "POST", body: JSON.stringify({ kind: "webcam", value: "0" }) });
    } catch (error) {
      notify((error as Error).message, "error");
    }
  };

  const useStream = async () => {
    if (!streamUrl.trim()) return;
    try {
      await api("/api/source", {
        method: "POST",
        body: JSON.stringify({ kind: "stream", value: streamUrl.trim() }),
      });
      setMenuOpen(false);
    } catch (error) {
      notify((error as Error).message, "error");
    }
  };

  const upload = async (file: File) => {
    setMenuOpen(false);
    setBusy(true);
    const form = new FormData();
    form.append("file", file);
    try {
      await api("/api/upload", { method: "POST", body: form });
    } catch (error) {
      notify((error as Error).message, "error");
    } finally {
      setBusy(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  };

  const enroll = async () => {
    if (!name.trim()) return;
    setBusy(true);
    try {
      await api("/api/enroll", { method: "POST", body: JSON.stringify({ name: name.trim() }) });
      setEnrollOpen(false);
      setName("");
    } catch (error) {
      notify((error as Error).message, "error");
    } finally {
      setBusy(false);
    }
  };

  const removeGallery = async (item: GalleryItem) => {
    try {
      await api(`/api/gallery/${item.id}`, { method: "DELETE" });
      notify(`已删除「${item.name}」`, "info");
    } catch (error) {
      notify((error as Error).message, "error");
    }
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (event.code === "Space") {
        event.preventDefault();
        control(pausedRef.current ? "resume" : "pause");
      } else if (event.code === "ArrowRight") {
        control("step");
      } else if (/^Digit[1-9]$/.test(event.code)) {
        const index = Number(event.code.slice(-1)) - 1;
        const face = facesRef.current[index];
        if (face) selectFace(face.id);
      } else if (event.key.toLowerCase() === "h") {
        control(holdRef.current ? "release" : "hold");
      } else if (event.key.toLowerCase() === "f") {
        toggleFullscreen();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [control, selectFace]);

  const toggleFullscreen = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(() => undefined);
  };

  const ratio = frame ? frame.w / frame.h : 16 / 9;
  const conf = selected?.face_conf ?? 0;
  const identity = selected?.identity;
  const identityName = identity?.status === "pending" ? "识别中…" : identity?.name ?? "未注册";
  const confColor = conf >= 0.82 ? "var(--green)" : conf >= 0.6 ? "var(--cyan)" : conf >= 0.4 ? "var(--amber)" : "var(--red)";

  return (
    <div className="screen live-screen">
      <header className="topbar">
        <div className="topbar-left">
          <Brand compact subtitle="观察者 · 左屏" />
          <span className="chip">{source?.label ?? "等待视频源"}</span>
        </div>
        <div className="topbar-right">
          <ConnectionPill connected={connected} />
          <div className="live-telemetry mono">
            <span><b>{stats?.fps?.toFixed(1) ?? "--"}</b> FPS</span>
            <span><b>{stats?.infer_ms?.toFixed(0) ?? "--"}</b> ms</span>
            <span><b>{faces.length}</b> 张脸</span>
          </div>
          <button className="btn btn-ghost btn-icon" onClick={() => control(paused ? "resume" : "pause")} title={paused ? "继续 (Space)" : "暂停 (Space)"}>
            {paused ? <Play size={16} /> : <Pause size={16} />}
          </button>
          <button className="btn btn-ghost btn-icon" onClick={() => control("step")} title="单步前进 (→)">
            <SkipForward size={16} />
          </button>
          <button
            className={`hold-btn ${hold ? "active" : ""}`}
            onClick={() => control(hold ? "release" : "hold")}
            disabled={!hold && !best}
            title={
              hold
                ? "解除最佳帧定格并继续推理 (H)"
                : best
                  ? `定格目前置信度最高的帧 ${pct(best.face_conf)} (H)`
                  : "还没有捕捉到带人脸的帧"
            }
          >
            <Trophy size={15} />
            <span>{hold ? `最佳帧 ${pct(hold.face_conf)}` : best ? `最佳 ${pct(best.face_conf, 0)}` : "定格最佳帧"}</span>
          </button>
          <div className="menu-anchor">
            <button className="btn" onClick={() => setMenuOpen((open) => !open)}>
              <Video size={15} /> 切换来源 <ChevronDown size={13} />
            </button>
            <AnimatePresence>
              {menuOpen && (
                <>
                  <div className="menu-scrim" onClick={() => setMenuOpen(false)} />
                  <motion.div
                    className="source-menu panel"
                    initial={{ opacity: 0, y: -8, scale: 0.98 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -6, scale: 0.98 }}
                    transition={{ duration: 0.16 }}
                  >
                    <div className="source-menu-title">视频来源</div>
                    <button className={`source-option ${source?.value === "closeup" && source.kind === "demo" ? "active" : ""}`} onClick={() => switchDemo("closeup")}>
                      <span className="source-icon"><ScanFace size={17} /></span>
                      <span><b>双人近景</b><em>2 张脸 · 身份识别演示</em></span>
                      {source?.value === "closeup" && source.kind === "demo" && <Check size={14} />}
                    </button>
                    <button className={`source-option ${source?.value === "multi" && source.kind === "demo" ? "active" : ""}`} onClick={() => switchDemo("multi")}>
                      <span className="source-icon"><Sparkles size={17} /></span>
                      <span><b>三人街景</b><em>多人同框 · 检测与追踪</em></span>
                      {source?.value === "multi" && source.kind === "demo" && <Check size={14} />}
                    </button>
                    <div className="source-divider" />
                    <button className="source-option" onClick={() => fileRef.current?.click()} disabled={busy}>
                      <span className="source-icon"><UploadCloud size={17} /></span>
                      <span><b>上传本地视频</b><em>mp4 / webm / mov · 最大 500MB</em></span>
                    </button>
                    <button className="source-option" onClick={useWebcam}>
                      <span className="source-icon"><Camera size={17} /></span>
                      <span><b>打开摄像头</b><em>本机无摄像头时请使用示例视频</em></span>
                    </button>
                    <div className="source-stream">
                      <div className="source-stream-label"><Link2 size={13} /> 视频流地址</div>
                      <div className="source-stream-row">
                        <input
                          value={streamUrl}
                          onChange={(event) => setStreamUrl(event.target.value)}
                          placeholder="rtsp:// 或 http://"
                          onKeyDown={(event) => event.key === "Enter" && useStream()}
                        />
                        <button className="btn btn-small" onClick={useStream}>连接</button>
                      </div>
                    </div>
                  </motion.div>
                </>
              )}
            </AnimatePresence>
          </div>
          <ThemeToggle />
          <button className="btn btn-ghost btn-icon" onClick={toggleFullscreen} title="全屏 (F)" aria-label="全屏">
            <Expand size={15} />
          </button>
        </div>
      </header>

      <main className="live-main">
        <section className="video-panel panel">
          <div className="video-toolbar">
            <div className="video-toolbar-left">
              <span className={`rec-badge ${hold ? "hold" : ""}`}><i /> {hold ? "BEST FRAME" : "LIVE FEED"}</span>
              <span className="video-title">{source?.label}</span>
              {source && <span className="chip">{source.kind === "demo" ? "DEMO LOOP" : source.kind.toUpperCase()}</span>}
              {hold && <span className="chip chip-amber">单帧分析 · #{hold.seq}</span>}
            </div>
            <div className="video-toolbar-right mono">
              {frame ? `${frame.w}×${frame.h}` : "WAITING FOR SIGNAL"}
              <span className="dot-sep" />
              YOLO26n-Pose
            </div>
          </div>

          <div className="video-fit">
            <div className={`video-frame ${hold ? "held" : ""}`} data-held={hold ? "1" : "0"} data-hold-seq={hold?.seq ?? ""} style={{ aspectRatio: `${ratio}`, ["--ar" as string]: ratio }}>
              {frame?.jpeg ? (
                <img src={`data:image/jpeg;base64,${frame.jpeg}`} alt="实时画面" draggable={false} />
              ) : (
                <div className="video-placeholder">
                  <Radar size={44} className="radar-icon" />
                  <span>正在建立实时数据流…</span>
                  <em>如果持续无画面，请确认后端已启动</em>
                </div>
              )}

              <div className="video-vignette" />
              <div className="scanline" />

              {faces.map((face) => (
                <FaceBox
                  key={face.id}
                  face={face}
                  selected={face.id === selected?.id}
                  onSelect={() => selectFace(face.id)}
                />
              ))}

              {selected && (
                <svg className="face-mesh" viewBox="0 0 100 100" preserveAspectRatio="none">
                  {FACE_KEYS.filter((index) => selected.keypoints[index][2] > 0.25).map((index) => {
                    const [x, y] = selected.keypoints[index];
                    return <circle key={index} cx={x * 100} cy={y * 100} r={index === 0 ? 0.55 : 0.65} />;
                  })}
                </svg>
              )}

              <div className="frame-corner tl" />
              <div className="frame-corner tr" />
              <div className="frame-corner bl" />
              <div className="frame-corner br" />

              <div className="video-hud left">
                <span className="hud-label">FACE CONFIDENCE</span>
                <b style={{ color: confColor }}>{selected ? pct(conf) : "--"}</b>
                <span className="hud-sub">{selected ? `TRACK #${String(selected.id).padStart(2, "0")}` : "NO FACE"}</span>
              </div>
              <div className="video-hud right">
                <span className="hud-label">{hold ? "BEST FRAME" : paused ? "PAUSED" : "STREAMING"}</span>
                <span className="hud-sub">{live ? new Date(live.t * 1000).toLocaleTimeString("zh-CN", { hour12: false }) : "--:--:--"}</span>
              </div>

              {!faces.length && frame?.jpeg && (
                <div className="no-face">
                  <div className="radar">
                    <i /><i /><i />
                  </div>
                  <div>
                    <b>等待人脸进入画面</b>
                    <span>人脸框由 17 个姿态关键点中的面部 5 点推导</span>
                  </div>
                </div>
              )}

              {paused && (
                hold ? (
                  <div className="paused-badge hold">
                    <Trophy size={13} /> 最佳帧定格 · {pct(hold.face_conf)} · PROBE #{hold.seq}
                  </div>
                ) : (
                  <div className="paused-badge">
                    <Pause size={13} /> 已暂停 · 按 → 单步
                  </div>
                )
              )}
            </div>
          </div>

          <div className="video-footer mono">
            <span><Wifi size={12} /> {connected ? "STREAM OK" : "RECONNECTING"}</span>
            <span><Clock3 size={12} /> IN {stats?.infer_ms?.toFixed(1) ?? "--"} ms</span>
            <span><Activity size={12} /> PROBE {stats?.probe_ms?.toFixed(0) ?? "--"} ms</span>
            <span className="grow" />
            <span>SPACE 暂停 · H 最佳帧 · → 单步 · 1–9 选脸 · F 全屏</span>
          </div>
        </section>

        <aside className="live-side">
          <div className="panel conf-panel">
            <div className="panel-head">
              <div className="panel-title"><ScanFace size={15} /> 人脸置信度</div>
              <span className="panel-kicker">FACE SCORE</span>
            </div>
            <div className="conf-body">
              <Gauge value={conf} size={168} />
              <div className="conf-breakdown">
                <ConfRow label="人物置信度" value={selected?.person_conf ?? 0} color="var(--blue)" />
                <ConfRow label="5 点关键点均值" value={selected?.kp_conf ?? 0} color="var(--cyan)" />
                <ConfRow label="眼部可见度 · 辅助" value={selected?.eye_conf ?? 0} color="var(--violet)" />
              </div>
            </div>
            <div className="conf-formula mono">
              <span>face = person × 5点均值</span>
              <b>{selected ? `${selected.person_conf.toFixed(2)} × ${selected.kp_conf.toFixed(2)} = ${conf.toFixed(3)}` : "等待检测"}</b>
            </div>
            <p className="conf-note">眼部可见度只用来判断对齐质量，不参与上面的乘积。</p>
            <div className="conf-spark">
              <div className="spark-head">
                <span>置信度走势</span>
                <span className="mono">{history.length ? `${history.length} 帧样本` : "暂无样本"}</span>
              </div>
              <Sparkline values={history} color={confColor} height={48} />
            </div>
          </div>

          <div className="panel identity-panel">
            <div className="panel-head">
              <div className="panel-title"><UserPlus size={15} /> 身份识别</div>
              <span className="panel-kicker">EDGEFACE-S · 512D</span>
            </div>
            <div className="identity-body">
              <div className={`identity-avatar ${identity?.status ?? "idle"}`}>
                {selected?.crop ? (
                  <img src={`data:image/jpeg;base64,${selected.crop}`} alt="对齐人脸" />
                ) : (
                  <ScanFace size={26} />
                )}
                {identity?.known && <span className="avatar-check"><Check size={10} /></span>}
              </div>
              <div className="identity-main">
                <div className="identity-name-row">
                  <b className={identity?.status === "pending" ? "shimmer-text" : ""}>{identityName}</b>
                  {identity?.status === "known" && <span className="chip chip-green">已匹配</span>}
                  {identity?.status === "unknown" && <span className="chip">未注册</span>}
                  {identity?.status === "pending" && <span className="chip chip-amber">比对中</span>}
                </div>
                <div className="sim-row">
                  <span>余弦相似度</span>
                  <b className="mono" style={{ color: identity?.known ? "var(--green)" : "var(--muted)" }}>
                    {identity?.similarity != null ? identity.similarity.toFixed(3) : "--"}
                  </b>
                </div>
                <div className="sim-bar">
                  <i style={{ width: `${Math.max(0, Math.min(1, identity?.similarity ?? 0)) * 100}%` }} />
                  <em style={{ left: `${(model?.match_threshold ?? 0.45) * 100}%` }} title="匹配阈值" />
                </div>
                <div className="sim-caption">
                  <span>{simLabel(identity?.similarity)}</span>
                  <span className="mono">阈值 {(model?.match_threshold ?? 0.45).toFixed(2)}</span>
                </div>
              </div>
            </div>

            <div className="identity-actions">
              {!enrollOpen ? (
                <button className="btn btn-small grow" onClick={() => setEnrollOpen(true)} disabled={!selected}>
                  <UserPlus size={13} /> 注册当前人脸
                </button>
              ) : (
                <div className="enroll-row">
                  <input autoFocus value={name} maxLength={24} placeholder="输入姓名" onChange={(event) => setName(event.target.value)} onKeyDown={(event) => event.key === "Enter" && enroll()} />
                  <button className="btn btn-small btn-primary" onClick={enroll} disabled={busy || !name.trim()}>
                    {busy ? <span className="spin"><Activity size={13} /></span> : <Check size={13} />}
                  </button>
                  <button className="btn btn-small btn-ghost btn-icon" onClick={() => setEnrollOpen(false)}><X size={13} /></button>
                </div>
              )}
              <span className="gallery-count mono">{gallery.length} 个档案</span>
            </div>

            {gallery.length > 0 && (
              <div className="gallery-strip">
                {gallery.map((item) => (
                  <div className="gallery-item" key={item.id} title={item.name}>
                    <img src={item.thumb} alt={item.name} />
                    <span>{item.name}</span>
                    <button onClick={() => removeGallery(item)} title={`删除 ${item.name}`}><Trash2 size={10} /></button>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="panel faces-panel">
            <div className="panel-head">
              <div className="panel-title"><Radar size={15} /> 画面中的人脸</div>
              <span className="panel-kicker">{faces.length} DETECTED</span>
            </div>
            <div className="face-list">
              {faces.length === 0 && (
                <div className="empty-state">
                  <WifiOff size={18} />
                  <span>当前画面没有检测到可用的人脸</span>
                </div>
              )}
              {faces.map((face, index) => (
                <button key={face.id} className={`face-row ${face.id === selected?.id ? "active" : ""}`} onClick={() => selectFace(face.id)}>
                  <span className="face-index mono">{String(index + 1).padStart(2, "0")}</span>
                  <span className="face-thumb">
                    {face.crop ? <img src={`data:image/jpeg;base64,${face.crop}`} alt={`人脸 ${face.id}`} /> : <ScanFace size={16} />}
                  </span>
                  <span className="face-row-main">
                    <span className="face-row-top">
                      <b>{face.identity.status === "pending" ? "识别中…" : face.identity.name}</b>
                      <em className="mono">{pct(face.face_conf)}</em>
                    </span>
                    <span className="mini-bar"><i style={{ width: `${face.face_conf * 100}%` }} /></span>
                    <span className="face-row-sub">
                      <span>#{String(face.id).padStart(2, "0")}</span>
                      <span>人物 {pct(face.person_conf, 0)}</span>
                      <span>{face.identity.similarity != null ? `相似 ${face.identity.similarity.toFixed(2)}` : "比对中"}</span>
                    </span>
                  </span>
                </button>
              ))}
            </div>
          </div>
        </aside>
      </main>

      <footer className="footer-bar">
        <div className="footer-metrics">
          <span><b>MODEL</b> {model?.name ?? "YOLO26n-Pose"} · {model?.runtime?.includes("TensorRT") ? "TensorRT" : "PyTorch"}</span>
          <span><b>INPUT</b> {model?.input ?? 640}×{model?.input ?? 640}</span>
          <span><b>GPU</b> {model?.gpu?.replace("Tesla ", "") ?? "detecting"}</span>
          <span><b>ID</b> {model?.embedder ?? "EdgeFace-S"}</span>
        </div>
        <div className="footer-metrics">
          <span><Cpu size={11} /> {model?.device?.toUpperCase() ?? "—"} · {stats?.gpu_mem_mb ? `${num(stats.gpu_mem_mb)} MB` : "本机推理"}</span>
          <span><Activity size={11} /> 探针每 {model?.probe_interval ?? 1.7}s</span>
          <span className="footer-note">画面只在本机处理，不上传云端</span>
        </div>
      </footer>

      <input
        ref={fileRef}
        type="file"
        accept="video/mp4,video/webm,video/quicktime,video/x-matroska,video/x-msvideo"
        hidden
        onChange={(event) => {
          const file = event.target.files?.[0];
          if (file) upload(file);
        }}
      />
      <Toasts toasts={toasts} dismiss={dismiss} />
    </div>
  );
}

function ConfRow({ label, value, color }: { label: string; value: number; color: string }) {
  return (
    <div className="conf-row">
      <div className="conf-row-head">
        <span>{label}</span>
        <b className="mono" style={{ color }}>{pct(value)}</b>
      </div>
      <div className="conf-row-bar"><i style={{ width: `${Math.min(1, Math.max(0, value)) * 100}%`, background: color }} /></div>
    </div>
  );
}

function FaceBox({
  face,
  selected,
  onSelect,
}: {
  face: FaceInfo;
  selected: boolean;
  onSelect: () => void;
}) {
  const [x1, y1, x2, y2] = face.box;
  const left = x1 * 100;
  const top = y1 * 100;
  const width = Math.max(0.5, (x2 - x1) * 100);
  const height = Math.max(0.5, (y2 - y1) * 100);
  const name = face.identity.status === "pending" ? "识别中…" : face.identity.name;
  const labelAbove = top > 9;

  return (
    <div
      className={`face-box ${selected ? "selected" : ""}`}
      style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}
      onClick={onSelect}
      role="button"
      tabIndex={0}
      onKeyDown={(event) => event.key === "Enter" && onSelect()}
    >
      <span className="fb-corner c1" />
      <span className="fb-corner c2" />
      <span className="fb-corner c3" />
      <span className="fb-corner c4" />
      <div className={`fb-label ${labelAbove ? "above" : "inside"}`}>
        <span className={`fb-dot ${face.identity.known ? "known" : ""}`} />
        <b>{name}</b>
        <span className="fb-conf mono">{pct(face.face_conf, 1)}</span>
        <span className="fb-id mono">#{String(face.id).padStart(2, "0")}</span>
      </div>
      {selected && <span className="fb-scan" />}
      {!selected && <span className="fb-mini-id mono">#{String(face.id).padStart(2, "0")}</span>}
      <span className="sr-only">{`人脸 ${face.id}，置信度 ${pct(face.face_conf)}`}</span>
    </div>
  );
}
