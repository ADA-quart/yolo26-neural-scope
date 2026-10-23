import { AnimatePresence, motion } from "framer-motion";
import {
  Activity,
  BrainCircuit,
  ChevronLeft,
  ChevronRight,
  Clock3,
  Expand,
  Eye,
  Gauge,
  Layers,
  Pause,
  Play,
  ScanFace,
  Sparkles,
  Timer,
  Trophy,
  Zap,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { Brand, ConnectionPill } from "../components/Brand";
import { ConvMicroscope } from "../components/neural/ConvMicroscope";
import { FeatureMap } from "../components/neural/FeatureMap";
import { HeadPanel } from "../components/neural/HeadPanel";
import { KernelOverlay } from "../components/neural/KernelOverlay";
import { LayerRail } from "../components/neural/LayerRail";
import { LayerStructure } from "../components/neural/LayerStructure";
import { StationBar } from "../components/neural/StationBar";
import { ThemeToggle } from "../components/ThemeToggle";
import { Toasts } from "../components/Toasts";
import { useFeed } from "../hooks/useFeed";
import { api, clamp, continuousProbe, num, pct, probeCadence } from "../lib/utils";
import { STATION_COLORS, STATIONS } from "../types";
import type { HoldInfo, MapEntry, MicroData, ProbeMsg } from "../types";

const TOUR = [0, 1, 2, 3, 5, 7, 10, 13, 16, 20, 22, 23];
const STATION_FOCUS = [0, 3, 13, 23];
const STATION_TINT = ["cyan", "blue", "violet", "amber"] as const;
const EYE_LAYERS = [2, 10, 22];

const TYPE_NOTES: Record<string, { look: string; meaning: string }> = {
  Conv: {
    look: "亮斑、条纹和色块——每个通道都在回应一种局部反差。",
    meaning: "卷积核在画面里寻找边缘、角点和颜色过渡，这是所有理解的起点。",
  },
  C3k2: {
    look: "比边缘更完整的局部图案，开始出现“像眼睛”“像鼻梁”的响应。",
    meaning: "多条卷积分支并行提取后合流，把零散线索组合成形状。",
  },
  SPPF: {
    look: "范围更大的响应块，小窗口、中窗口和大窗口的结果叠在一起。",
    meaning: "同一位置被多种尺度同时观察，小脸和大脸都能被看见。",
  },
  C2PSA: {
    look: "重点区域被点亮，背景响应被压低。",
    meaning: "注意力机制重新分配权重，让真正重要的位置进入后面的决策。",
  },
  Upsample: {
    look: "颗粒更细的放大图，语义信息被铺回较大的网格。",
    meaning: "把深层的“看懂”放大，准备和浅层的“看清”对齐。",
  },
  Concat: {
    look: "不同来源的通道并排放在一起，细节和语义同时存在。",
    meaning: "把浅层细节与深层语义拼接，后续卷积可以同时参考两者。",
  },
  Pose26: {
    look: "300 个槽位排成一片，只有极少数亮起——每个亮格就是一个正在给出结果的查询。",
    meaning: "每个槽位输出框、置信度、类别和 17 个关键点；一对一匹配让每个目标只留一个框，所以不需要 NMS。",
  },
};

const pad2 = (value: number) => String(value).padStart(2, "0");

export default function NeuralScreen() {
  const feed = useFeed();
  const { live, probe, state, model, stats, connected, toasts, notify, dismiss } = feed;
  const [focus, setFocus] = useState(0);
  const [touring, setTouring] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [engineer, setEngineer] = useState(false);
  const [maps, setMaps] = useState<Record<number, MapEntry>>({});
  const [micro, setMicro] = useState<Record<number, MicroData>>({});
  const [microStep, setMicroStep] = useState(0);
  const [microPlaying, setMicroPlaying] = useState(true);
  const [microLocked, setMicroLocked] = useState(false);
  const [lastProbe, setLastProbe] = useState<ProbeMsg | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const seenProbe = useRef(-1);
  const lastFlashAt = useRef(0);
  const holdRef = useRef(false);
  const sourceKey = live ? `${live.source.kind}:${live.source.value}:${live.source.start}:${live.source.end}` : null;
  const lastSourceKey = useRef<string | null>(null);

  const meta = model?.meta ?? [];
  const layer = meta.find((item) => item.i === focus) ?? meta[0];
  const prevLayer = meta.find((item) => item.i === focus - 1);
  const station = layer?.station ?? 0;
  const layerMicro = micro[focus];
  const paused = state?.paused ?? false;
  const hold = state?.hold ?? null;
  const best = stats?.best ?? null;
  const probingContinuously = continuousProbe(model?.probe_interval, model?.probe_continuous);
  holdRef.current = !!hold;
  const analysisFrame = lastProbe?.frame ?? live?.frame;
  const previewFrame = live?.frame ?? lastProbe?.frame;
  const previewFaces = live?.faces ?? lastProbe?.faces ?? [];
  const selected =
    previewFaces.find((face) => face.selected) ??
    previewFaces.find((face) => face.id === (live?.selected ?? lastProbe?.face?.id ?? state?.selected)) ??
    previewFaces[0] ??
    null;
  const identity = selected?.identity;
  const note = TYPE_NOTES[layer?.type ?? "Conv"] ?? TYPE_NOTES.Conv;
  const mapCount = Object.values(maps).filter((entry) => entry.map).length;
  const microCount = Object.keys(micro).length;
  const microLayer = layerMicro?.layer ?? null;
  const microStepCount = layerMicro?.steps.length ?? 0;
  const safeMicroStep = Math.min(microStep, Math.max(0, microStepCount - 1));
  const forwardSeq = lastProbe?.seq ?? null;

  // 右屏只展示探针那一次前向的结果，避免把新的实时帧套在旧特征图上。
  const head = lastProbe?.head ?? null;

  useEffect(() => {
    if (!sourceKey) return;
    if (lastSourceKey.current === null) {
      lastSourceKey.current = sourceKey;
      return;
    }
    if (lastSourceKey.current === sourceKey) return;
    lastSourceKey.current = sourceKey;
    setLastProbe(null);
    setMaps({});
    setMicro({});
    seenProbe.current = -1;
  }, [sourceKey]);

  useEffect(() => {
    if (!probe || probe.seq === seenProbe.current) return;
    seenProbe.current = probe.seq;
    setLastProbe(probe);
    const now = Date.now();
    if (now - lastFlashAt.current >= 900) {
      lastFlashAt.current = now;
      setRefreshKey((value) => value + 1);
    }
    setMaps((prev) => {
      const next = { ...prev };
      for (const entry of probe.maps) next[entry.i] = entry;
      return next;
    });
    setMicro((prev) => {
      const next = { ...prev };
      for (const [key, value] of Object.entries(probe.micro)) next[Number(key)] = value;
      return next;
    });
  }, [probe]);

  useEffect(() => {
    setMicroStep(0);
    setMicroPlaying(true);
    setMicroLocked(false);
  }, [microLayer]);

  useEffect(() => {
    if (microStepCount <= 1 || !microPlaying || microLocked) return;
    const timer = window.setInterval(() => {
      setMicroStep((value) => (value + 1) % microStepCount);
    }, 720);
    return () => window.clearInterval(timer);
  }, [microStepCount, microPlaying, microLocked]);

  const move = useCallback((direction: number) => {
    setFocus((current) => {
      const index = TOUR.indexOf(current);
      if (index < 0) return TOUR[0];
      return TOUR[(index + direction + TOUR.length) % TOUR.length];
    });
  }, []);

  const toggleHold = useCallback(async () => {
    try {
      await api("/api/control", {
        method: "POST",
        body: JSON.stringify({ action: holdRef.current ? "release" : "hold" }),
      });
    } catch (error) {
      notify((error as Error).message, "error");
    }
  }, [notify]);

  useEffect(() => {
    if (!touring) return;
    const dwell = (focus === 23 ? 2600 : 1450) / speed;
    const timer = window.setTimeout(() => move(1), dwell);
    return () => window.clearTimeout(timer);
  }, [focus, touring, speed, move]);

  const pickLayer = (index: number) => {
    setTouring(false);
    setFocus(index);
  };

  const pickStation = (index: number) => {
    setTouring(false);
    setFocus(STATION_FOCUS[index] ?? 0);
  };

  const toggleFullscreen = () => {
    if (document.fullscreenElement) document.exitFullscreen();
    else document.documentElement.requestFullscreen().catch(() => undefined);
  };

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      if (target && ["INPUT", "TEXTAREA", "SELECT"].includes(target.tagName)) return;
      if (event.code === "Space") {
        event.preventDefault();
        setTouring((value) => !value);
      } else if (event.code === "ArrowLeft") {
        event.preventDefault();
        setTouring(false);
        move(-1);
      } else if (event.code === "ArrowRight") {
        event.preventDefault();
        setTouring(false);
        move(1);
      } else if (/^Digit[1-4]$/.test(event.code)) {
        pickStation(Number(event.code.slice(-1)) - 1);
      } else if (event.key.toLowerCase() === "h") {
        toggleHold();
      } else if (event.key.toLowerCase() === "e") {
        setEngineer((value) => !value);
      } else if (event.key.toLowerCase() === "f") {
        toggleFullscreen();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [move, toggleHold]);

  const tint = STATION_TINT[station] ?? "cyan";
  const ratio = previewFrame ? `${previewFrame.w} / ${previewFrame.h}` : "16 / 9";
  const identityName = identity?.status === "pending" ? "识别中…" : identity?.name ?? "未注册";

  return (
    <div className="screen neural-screen">
      <header className="topbar neural-topbar">
        <div className="topbar-left">
          <Brand compact subtitle="神经显微镜 · 右屏" />
          <span className="chip">{live?.source.label ?? state?.source.label ?? "等待视频源"}</span>
          <span className={`chip ${hold || paused ? "chip-amber" : "chip-green"}`}>
            {hold ? `最佳帧 · ${pct(hold.face_conf)}` : paused ? "推理已暂停" : "实时推理"}
          </span>
        </div>
        <div className="topbar-right">
          <div className={`tour-controls ${touring ? "playing" : ""}`}>
            <span className="tour-label mono">自动导览</span>
            <button className="tour-btn" onClick={() => { setTouring(false); move(-1); }} title="上一层 (←)">
              <ChevronLeft size={14} />
            </button>
            <button className="tour-btn primary" onClick={() => setTouring((value) => !value)} title={touring ? "暂停导览 (Space)" : "开始导览 (Space)"}>
              {touring ? <Pause size={13} /> : <Play size={13} />}
            </button>
            <button className="tour-btn" onClick={() => { setTouring(false); move(1); }} title="下一层 (→)">
              <ChevronRight size={14} />
            </button>
            <div className="speed-switch mono">
              {[0.75, 1, 1.5].map((value) => (
                <button key={value} className={speed === value ? "active" : ""} onClick={() => setSpeed(value)}>
                  {value}×
                </button>
              ))}
            </div>
          </div>
          <div className="mode-switch">
            <button className={!engineer ? "active" : ""} onClick={() => setEngineer(false)}>新手</button>
            <button className={engineer ? "active" : ""} onClick={() => setEngineer(true)}>工程师</button>
          </div>
          <button
            className={`hold-btn neural-hold ${hold ? "active" : ""}`}
            onClick={toggleHold}
            disabled={!hold && !best}
            title={
              hold
                ? "解除最佳帧定格并继续推理 (H)"
                : best
                  ? `定格目前置信度最高的帧 ${pct(best.face_conf)}，暂停后逐层分析这一帧 (H)`
                  : "还没有捕捉到带人脸的帧"
            }
          >
            <Trophy size={13} />
            <span>{hold ? `已定格 ${pct(hold.face_conf, 0)}` : best ? `最佳 ${pct(best.face_conf, 0)}` : "最佳帧"}</span>
          </button>
          <ConnectionPill connected={connected} />
          <ThemeToggle />
          <button className="btn btn-ghost btn-icon" onClick={toggleFullscreen} title="全屏 (F)" aria-label="全屏">
            <Expand size={15} />
          </button>
        </div>
      </header>

      <main className="neural-main">
        <StationBar active={station} focus={focus} onPick={pickStation} live={!!lastProbe} />

        <div className="n-grid">
          <aside className="n-col n-col-left">
            <section className="panel n-input-card">
              <div className="panel-head">
                <div className="panel-title"><ScanFace size={15} /> 实时取景</div>
                <span className="panel-kicker">
                  {hold
                    ? "HELD BEST · 单帧"
                    : forwardSeq != null
                      ? `LIVE PREVIEW · 分析帧 #${forwardSeq}`
                      : "LIVE PREVIEW"}
                </span>
              </div>
              <div className="n-input" data-probe-seq={forwardSeq ?? ""} data-held={hold ? "1" : "0"} data-hold-seq={hold?.seq ?? ""} style={{ aspectRatio: ratio }}>
                {previewFrame?.jpeg ? (
                  <img src={`data:image/jpeg;base64,${previewFrame.jpeg}`} alt="实时取景画面" draggable={false} />
                ) : (
                  <div className="n-input-empty"><ScanFace size={30} /><span>等待实时画面…</span></div>
                )}
                <div className="scanline" />
                {previewFaces.map((face) => (
                  <span
                    key={face.id}
                    className={`n-face ${face.id === selected?.id ? "sel" : ""}`}
                    style={{
                      left: `${face.box[0] * 100}%`,
                      top: `${face.box[1] * 100}%`,
                      width: `${(face.box[2] - face.box[0]) * 100}%`,
                      height: `${(face.box[3] - face.box[1]) * 100}%`,
                    }}
                  />
                ))}
                {selected && (
                  <>
                    <span
                      className={`n-roi-tag mono ${selected.box[1] < 0.1 ? "inside" : ""}`}
                      style={{ left: `${selected.box[0] * 100}%`, top: `${selected.box[1] * 100}%` }}
                    >
                      ROI · #{pad2(selected.id)}
                    </span>
                    <div className="n-input-hud">
                      {selected.crop && <img src={`data:image/jpeg;base64,${selected.crop}`} alt="选中人脸" />}
                      <div>
                        <b>{identityName}</b>
                        <span className="mono">
                          {pct(selected.face_conf)} · {identity?.similarity != null ? `相似 ${identity.similarity.toFixed(2)}` : "比对中"}
                        </span>
                      </div>
                      <span className={`chip ${identity?.status === "known" ? "chip-green" : identity?.status === "pending" ? "chip-amber" : ""}`}>
                        {identity?.status === "known" ? "已匹配" : identity?.status === "pending" ? "比对中" : "未注册"}
                      </span>
                    </div>
                  </>
                )}
              </div>
              <div className="n-input-meta">
                <span className="mono">{previewFrame ? `${previewFrame.w}×${previewFrame.h}` : "--"}</span>
                <span className="mono">{forwardSeq != null ? `PROBE #${forwardSeq}` : "等待探针"}</span>
                <span className="mono">{hold ? "已定格" : "实时"}</span>
                <span className="n-input-source">{live?.source.label ?? state?.source.label ?? "--"}</span>
              </div>
              <p className="n-input-note">
                {hold
                  ? `这一帧是按最高人脸置信度挑出的最佳现场，推理已暂停；左边的层图、右边的取景窗和决策头仍是同一次前向，可以逐层慢慢看。按 H 或右上角按钮继续实时。`
                  : probingContinuously
                    ? "取景窗跟随视频源实时刷新；后台探针完成一次完整前向后立刻接下一帧，层图和决策头因此几乎跟着画面走，并且始终来自同一次前向。"
                    : `取景窗跟随视频源实时刷新；层图和决策头来自最近一次探针——每 ${model?.probe_interval ?? 0.6} 秒抓取一帧做完整前向，所以分析结果比画面慢一步，但彼此严格同帧。`}
              </p>
            </section>

            <section className="panel n-eye-card">
              <div className="panel-head">
                <div className="panel-title"><Eye size={15} /> 网络之眼</div>
                <span className="panel-kicker">SAME PROBE · 3 DEPTHS</span>
              </div>
              <div className="eye-grid">
                {EYE_LAYERS.map((index, order) => {
                  const eyeLayer = meta.find((item) => item.i === index);
                  return (
                    <button className={`eye-cell ${focus === index ? "active" : ""}`} key={index} onClick={() => pickLayer(index)}>
                      <FeatureMap entry={maps[index]} tint={STATION_TINT[eyeLayer?.station ?? 0]} showScan={false} badge={`L${pad2(index)}`} />
                      <span className="eye-caption">
                        <b>{order === 0 ? "浅层" : order === 1 ? "中层" : "深层"}</b>
                        <em className="mono">{eyeLayer?.shape?.slice(1).join("×") ?? "--"}</em>
                      </span>
                    </button>
                  );
                })}
              </div>
              <p className="eye-note">同一帧画面在三个深度上的真实响应：越往下，分辨率越低、语义越抽象，最后交给决策头给出框和关键点。</p>
            </section>

            <section className="panel n-telemetry">
              <div className="panel-head">
                <div className="panel-title"><Gauge size={15} /> 本机遥测</div>
                <span className="panel-kicker">LIVE TELEMETRY</span>
              </div>
              <div className="n-kv">
                <div><span>画面帧率</span><b className="mono">{stats?.fps?.toFixed(1) ?? "--"} <em>FPS</em></b></div>
                <div><span>单次前向</span><b className="mono">{stats?.infer_ms?.toFixed(1) ?? "--"} <em>ms · {stats?.runtime === "tensorrt" ? "TRT" : "PT"}</em></b></div>
                <div><span>完整探针</span><b className="mono">{stats?.probe_ms?.toFixed(0) ?? "--"} <em>ms</em></b></div>
                <div><span>显存占用</span><b className="mono">{stats?.gpu_mem_mb != null ? `${num(stats.gpu_mem_mb)} MB` : "--"}</b></div>
                <div><span>模型参数</span><b className="mono">{model ? `${(model.params / 1_000_000).toFixed(2)} M` : "--"}</b></div>
                <div><span>识别后端</span><b className="mono n-kv-small">{model?.embedder ?? "--"}</b></div>
              </div>
              <div className="n-telemetry-foot mono">
                <Zap size={11} /> {model?.gpu?.replace("Tesla ", "") ?? "GPU"} · {model?.device?.toUpperCase() ?? "--"} · CUDA {model?.cuda ?? "--"} · {model?.runtime ?? "PyTorch"}
              </div>
            </section>
          </aside>

          <section className={`n-stage ${hold ? "held" : ""}`} data-held={hold ? "1" : "0"} data-hold-seq={hold?.seq ?? ""}>
            <div className="n-stage-head panel">
              <div className="n-stage-top">
                <div className="n-stage-id">
                  <span className="n-layer-no mono" style={{ ["--station" as string]: STATION_COLORS[station] }}>
                    L{pad2(focus)}
                  </span>
                  <div className="n-stage-copy">
                    <div className="n-stage-title">{layer?.caption_zh ?? "等待模型元数据"}</div>
                    <div className="n-stage-sub mono">
                      {layer?.group ?? "--"} · {layer?.type ?? "--"}
                      {layer?.shape ? ` · ${layer.shape.slice(1).join(" × ")}` : ""}
                    </div>
                  </div>
                </div>
                <div className="n-stage-chips mono">
                  {hold && <span className="n-held-chip">最佳帧 · {pct(hold.face_conf)}</span>}
                  {forwardSeq != null && <span className="n-forward-chip">同帧前向 #{forwardSeq}</span>}
                  {layer?.type === "Pose26" ? (
                    <span>300 槽位 × 57 个数</span>
                  ) : layer?.kernel ? (
                    <span>K{layer.kernel} · S{layer.stride} · P{layer.padding}</span>
                  ) : null}
                  {layer?.params ? <span>{num(layer.params)} params</span> : null}
                  {layer?.gpu_ms != null ? <span>{layer.gpu_ms.toFixed(2)} ms</span> : null}
                </div>
              </div>
              <p className="n-stage-explain">
                <span className="mono">为什么看这里 ·</span>{" "}
                {engineer ? layer?.detail_zh : note.meaning}
              </p>
            </div>

            <AnimatePresence mode="wait" initial={false}>
              <motion.div
                key={focus}
                className="n-stage-body"
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0, y: -6 }}
                transition={{ duration: 0.18 }}
              >
                {focus === 23 ? (
                  <HeadPanel full head={head} frame={analysisFrame} seq={forwardSeq} />
                ) : (
                  <>
                    <div className="n-map-wrap" data-probe-seq={forwardSeq ?? ""}>
                      <FeatureMap
                        entry={maps[focus]}
                        className="n-stage-map"
                        tint={tint}
                        badge={maps[focus]?.map ? "真实层图 · 3 通道合成" : "等待探针"}
                      />
                      <span key={refreshKey} className="n-map-flash" />
                      {layerMicro && (
                        <KernelOverlay
                          data={layerMicro}
                          step={safeMicroStep}
                          seq={forwardSeq}
                          playing={microPlaying}
                          onTogglePlay={() => setMicroPlaying((value) => !value)}
                        />
                      )}
                      {engineer && (
                        <div className="n-map-readout mono">
                          <span>ACT {maps[focus]?.activity != null ? pct(maps[focus].activity, 1) : "--"}</span>
                          <span>{layer?.params ? `${num(layer.params)} params` : ""}</span>
                          <span>{layer?.gpu_ms != null ? `${layer.gpu_ms.toFixed(2)} ms` : ""}</span>
                        </div>
                      )}
                    </div>
                    {layerMicro ? (
                      <ConvMicroscope
                        data={layerMicro}
                        layer={layer}
                        engineer={engineer}
                        step={safeMicroStep}
                        playing={microPlaying}
                        locked={microLocked}
                        onStep={(index) => {
                          setMicroStep(index);
                          setMicroPlaying(false);
                        }}
                        onTogglePlay={() => setMicroPlaying((value) => !value)}
                        onHover={setMicroLocked}
                      />
                    ) : layer ? (
                      <LayerStructure layer={layer} prevShape={prevLayer?.shape} />
                    ) : null}
                  </>
                )}
              </motion.div>
            </AnimatePresence>

            {focus !== 23 && (
              <div className="n-stage-note panel">
                <span className="n-stage-note-icon"><Eye size={13} /></span>
                <p>
                  <b>怎么读这张图：</b>
                  {engineer
                    ? `取整张层图上方差最大的 3 个通道，各自 min-max 归一化后叠成 RGB；活度是归一化图像的平均亮度。${
                        layerMicro
                          ? "ROI 决定左侧取景窗和右侧显微镜取哪块输入；取景窗底图是卷积输入通道，不是这张输出层图。"
                          : ""
                      }`
                    : `亮的地方 = 这个通道在这一位置反应强。颜色不是原图：系统挑了整帧里变化最明显的 3 个通道，分别归一化后叠成 RGB。${
                        layerMicro
                          ? "左上角取景窗的底图是卷积真正读入的输入通道，和右侧乘加同步滑动，不是这张输出图本身。"
                          : ""
                      }`}
                </p>
                <span className="n-stage-note-tag mono">REAL FORWARD · L{pad2(focus)}</span>
              </div>
            )}
          </section>

          <aside className="n-col n-col-right">
            <section className="panel n-head-card">
              <div className="panel-head">
                <div className="panel-title"><BrainCircuit size={15} /> 决策头 · 这一帧的答案</div>
                <button className="btn btn-mini" onClick={() => pickLayer(23)}>展开决策头</button>
              </div>
              <HeadPanel head={head} frame={analysisFrame} seq={forwardSeq} />
            </section>

            <section className="panel n-why-card">
              <div className="panel-head">
                <div className="panel-title"><Layers size={15} /> 这一层在做什么</div>
                <span className="panel-kicker">{STATIONS[station]?.en ?? "--"}</span>
              </div>
              <div className="n-why-station" style={{ ["--station" as string]: STATION_COLORS[station] }}>
                <span className="n-why-node"><i /></span>
                <div>
                  <b>{STATIONS[station]?.name} · {STATIONS[station]?.en}</b>
                  <span>{STATIONS[station]?.hint}</span>
                </div>
              </div>
              <div className="n-why-block">
                <span className="n-why-label"><Eye size={11} /> 画面里能看见什么</span>
                <p>{note.look}</p>
              </div>
              <div className="n-why-block">
                <span className="n-why-label"><Sparkles size={11} /> 对识别意味着什么</span>
                <p>{note.meaning}</p>
              </div>
              {engineer && layer && (
                <div className="n-prop-grid mono">
                  <div><span>shape</span><b>{layer.shape?.join("×") ?? "--"}</b></div>
                  <div><span>kernel</span><b>{layer.kernel ?? "—"}</b></div>
                  <div><span>stride</span><b>{layer.stride ?? "—"}</b></div>
                  <div><span>padding</span><b>{layer.padding ?? "—"}</b></div>
                  <div><span>groups</span><b>{layer.groups ?? "—"}</b></div>
                  <div><span>params</span><b>{num(layer.params)}</b></div>
                </div>
              )}
              <div className="n-route">
                {STATIONS.map((item, index) => (
                  <button
                    key={item.id}
                    className={`n-route-step ${index === station ? "active" : ""} ${index < station ? "done" : ""}`}
                    onClick={() => pickStation(index)}
                    style={{ ["--station" as string]: STATION_COLORS[index] }}
                    title={`跳转到「${item.name}」`}
                  >
                    <i />
                    <span>{item.name}</span>
                  </button>
                ))}
              </div>
            </section>

            <ProbeStatus
              probe={lastProbe}
              paused={paused}
              hold={hold}
              mapCount={mapCount}
              microCount={microCount}
              interval={model?.probe_interval ?? 0.6}
              continuous={probingContinuously}
            />
          </aside>
        </div>

        <div className="panel n-rail-panel">
          <div className="n-rail-head">
            <span className="n-rail-title"><Layers size={13} /> 24 个顶层模块 · 23 张真实层图 + 决策头</span>
            <span className="n-rail-hint mono">
              {maps[focus] ? `层 ${pad2(focus)} 已有数据` : "等待探针"} · 点击任意层接管导览
            </span>
          </div>
          <LayerRail meta={meta} maps={maps} focus={focus} onSelect={pickLayer} fresh={!!lastProbe} />
        </div>
      </main>

      <footer className="footer-bar">
        <div className="footer-metrics">
          <span><b>MODEL</b> {model?.name ?? "YOLO26n-Pose"}</span>
          <span><b>TRACK</b> #{selected ? pad2(selected.id) : "--"}</span>
          <span><b>FOCUS</b> L{pad2(focus)} · {layer?.type ?? "--"}</span>
          <span><b>MAPS</b> {mapCount}/23 · 卷积显微镜 {microCount}</span>
        </div>
        <div className="footer-metrics">
          <span>
            <Timer size={11} /> {probeCadence(model?.probe_interval, model?.probe_continuous)}
            {probingContinuously ? ` · ${lastProbe?.took_ms.toFixed(0) ?? "--"} ms/次` : ""}
          </span>
          <span><Activity size={11} /> SPACE 导览 · ←/→ 换层 · 1–4 站点 · H 最佳帧 · E 模式 · F 全屏</span>
          <span className="footer-note">
            {hold
              ? `最佳帧 #${hold.seq} · 置信度 ${pct(hold.face_conf)} · 推理已暂停`
              : lastProbe
                ? "取景窗实时刷新 · 23 张层图与决策头同帧"
                : "等待第一次完整探针"}
          </span>
        </div>
      </footer>

      <Toasts toasts={toasts} dismiss={dismiss} />
    </div>
  );
}

function ProbeStatus({
  probe,
  paused,
  hold,
  mapCount,
  microCount,
  interval,
  continuous,
}: {
  probe: ProbeMsg | null;
  paused: boolean;
  hold: HoldInfo | null;
  mapCount: number;
  microCount: number;
  interval: number;
  continuous: boolean;
}) {
  const [now, setNow] = useState(() => Date.now() / 1000);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 250);
    return () => window.clearInterval(timer);
  }, []);

  const anchor = hold?.t ?? probe?.t ?? null;
  const age = anchor != null ? Math.max(0, now - anchor) : null;
  const ageText = age == null ? "--" : age < 60 ? `${age.toFixed(1)}s` : `${Math.floor(age / 60)}m${Math.round(age % 60)}s`;
  const progress = hold || continuous ? 0 : age == null ? 0 : clamp(age / interval, 0, 1);

  return (
    <section className={`panel n-probe-card ${hold ? "held" : ""}`}>
      <div className="n-probe-top">
        <span className={`n-probe-dot ${probe ? "live" : ""} ${paused ? "paused" : ""} ${hold ? "held" : ""}`} />
        <b>{hold ? "最佳帧已定格 · 单帧分析" : paused ? "探针已暂停" : probe ? "探针持续采样" : "等待第一次探针"}</b>
        <span className="mono">{hold ? `捕获于 ${ageText} 前` : `${ageText} 前`}</span>
      </div>
      <div className="n-probe-bar"><i style={{ width: `${(1 - progress) * 100}%` }} /></div>
      <div className="n-probe-metrics mono">
        <span><Clock3 size={10} /> {probe ? `${probe.took_ms.toFixed(0)} ms` : "--"}</span>
        <span><Layers size={10} /> {mapCount}/23 MAPS</span>
        <span><Zap size={10} /> {microCount} CONV</span>
      </div>
      <p>
        {hold
          ? `已冻结在置信度 ${pct(hold.face_conf)} 的最佳现场。层图、卷积取景窗和决策头都属于这一帧，可以继续逐层检查；按 H 或右上角按钮恢复实时。`
          : continuous
            ? `探针连续运行：单次完整前向约 ${probe?.took_ms.toFixed(0) ?? "--"} ms，完成即接下一帧；这一帧的输入图、层图和决策头一起更新，左侧观察窗仍按实时帧率播放。`
            : `每 ${interval.toFixed(1)} 秒挑一帧做完整探针：这一帧的输入图、层图和决策头会一起更新；左侧观察窗仍按实时帧率播放。`}
      </p>
    </section>
  );
}
