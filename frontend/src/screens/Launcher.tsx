import { motion } from "framer-motion";
import {
  ArrowRight,
  ArrowUpRight,
  Cpu,
  Eye,
  Laptop,
  Layers3,
  Monitor,
  Network,
  Play,
  ScanFace,
  Sparkles,
  UserCheck,
  Users,
} from "lucide-react";
import { useMemo } from "react";
import { Brand, ConnectionPill } from "../components/Brand";
import { ThemeToggle } from "../components/ThemeToggle";
import { Toasts } from "../components/Toasts";
import { useFeed } from "../hooks/useFeed";
import { continuousProbe, openWindow, pct } from "../lib/utils";
import { STATIONS, STATION_COLORS } from "../types";

const fadeUp = {
  hidden: { opacity: 0, y: 18 },
  show: (delay: number) => ({
    opacity: 1,
    y: 0,
    transition: { duration: 0.6, delay, ease: [0.22, 0.8, 0.3, 1] as const },
  }),
};

export default function Launcher() {
  const feed = useFeed();
  const { live, probe, head, model, state, stats, connected, toasts, notify, dismiss } = feed;

  const face = live?.faces?.find((item) => item.selected) ?? live?.faces?.[0] ?? null;
  const previewMap = useMemo(() => {
    if (!probe?.maps?.length) return null;
    const preferred = [7, 3, 10, 5, 0].map((index) => probe.maps.find((item) => item.i === index && item.map));
    return preferred.find(Boolean) ?? probe.maps.find((item) => item.map) ?? null;
  }, [probe]);
  const headData = probe?.head ?? head?.head ?? null;

  const openBoth = () => {
    const left = openWindow("/live", "yolo26-live", 1280, 820);
    const right = openWindow("/neural", "yolo26-neural", 1600, 940);
    if (!left || !right) {
      notify("浏览器拦截了弹出窗口。请允许弹窗，或分别点击「左屏」「右屏」按钮。", "warn");
    }
  };

  return (
    <div className="launcher">
      <header className="landing-nav">
        <Brand subtitle="YOLO26 实时视觉实验室" />
        <div className="landing-nav-right">
          <ConnectionPill connected={connected} />
          <span className="chip">{model?.gpu?.replace("Tesla ", "") ?? "正在连接引擎"}</span>
          <span className="chip">{(state?.gallery.length ?? 0)} 个身份档案</span>
          <ThemeToggle />
          <button className="btn btn-primary btn-small" onClick={openBoth}>
            <Play size={13} /> 打开双窗
          </button>
        </div>
      </header>

      <main className="landing-main">
        <section className="hero">
          <motion.div className="hero-copy" variants={fadeUp} initial="hidden" animate="show" custom={0.05}>
            <div className="eyebrow">REAL-TIME VISION LAB · YOLO26</div>
            <h1>
              看见 YOLO
              <span className="gradient-text">如何思考</span>
            </h1>
            <p className="hero-lead">
              左边，是一块认真工作的观察屏：实时框出每一张脸，给出人脸置信度，并说出「这是谁」。
              右边，是它的神经显微镜：真实卷积、真实特征图、真实 300 个查询槽位，一层一层拆给你看。
            </p>
            <div className="hero-actions">
              <button className="btn btn-primary btn-hero" onClick={openBoth}>
                <Laptop size={16} />
                打开双屏体验
                <ArrowRight size={16} />
              </button>
              <button className="btn btn-hero" onClick={() => openWindow("/live", "yolo26-live", 1280, 820)}>
                <ScanFace size={15} /> 只开左屏
              </button>
              <button className="btn btn-ghost btn-hero" onClick={() => openWindow("/neural", "yolo26-neural", 1600, 940)}>
                <Network size={15} /> 只开右屏
              </button>
            </div>
            <div className="hero-note">
              左屏放笔记本，右屏拖到外接显示器，按 F11 全屏。窗口被拦截时，请允许此站点弹出窗口。
            </div>

            <div className="hero-stats">
              <div><b>24</b><span>顶层模块 · 23 层图</span></div>
              <div><b>300</b><span>查询槽位</span></div>
              <div><b>512D</b><span>人脸特征</span></div>
              <div><b>{stats?.fps?.toFixed(1) ?? "--"}</b><span>当前 FPS</span></div>
            </div>
          </motion.div>

          <motion.div className="display-stage" variants={fadeUp} initial="hidden" animate="show" custom={0.22}>
            <div className="stage-glow" />
            <div className="laptop-wrap">
              <div className="screen-label">
                <span><Laptop size={12} /> 左屏 · 笔记本</span>
                <em>观察者</em>
              </div>
              <div className="laptop">
                <div className="laptop-screen">
                  <div className="mock-topbar">
                    <span className="mock-brand"><i /> YOLO26 · 观察者</span>
                    <span className="mock-live"><i /> LIVE</span>
                  </div>
                  <div className="mock-video">
                    {live?.frame?.jpeg ? (
                      <img src={`data:image/jpeg;base64,${live.frame.jpeg}`} alt="实时画面预览" />
                    ) : (
                      <div className="mock-placeholder">
                        <ScanFace size={30} />
                        <span>正在等待实时画面</span>
                      </div>
                    )}
                    {face && live?.frame && (
                      <span
                        className="mock-facebox"
                        style={{
                          left: `${face.box[0] * 100}%`,
                          top: `${face.box[1] * 100}%`,
                          width: `${(face.box[2] - face.box[0]) * 100}%`,
                          height: `${(face.box[3] - face.box[1]) * 100}%`,
                        }}
                      >
                        <em>{face.identity.name}</em>
                      </span>
                    )}
                    <div className="mock-video-hud">
                      <span>FACE CONFIDENCE</span>
                      <b>{face ? pct(face.face_conf) : "--"}</b>
                    </div>
                  </div>
                  <div className="mock-side">
                    <div className="mock-gauge" style={{ ["--conf" as string]: face?.face_conf ?? 0 }}>
                      <b>{face ? (face.face_conf * 100).toFixed(0) : "--"}</b>
                      <span>置信度</span>
                    </div>
                    <div className="mock-rows">
                      <span><i style={{ width: `${(face?.face_conf ?? 0) * 100}%` }} /></span>
                      <span><i style={{ width: `${(face?.identity.similarity ?? 0) * 100}%` }} /></span>
                      <span><i style={{ width: `${Math.max(8, (face?.kp_conf ?? 0) * 100)}%` }} /></span>
                    </div>
                    <div className="mock-identity">
                      <b>{face?.identity.name ?? "未检测"}</b>
                      <em>ID 识别</em>
                    </div>
                  </div>
                </div>
                <div className="laptop-base" />
              </div>
            </div>

            <div className="monitor-wrap">
              <div className="screen-label">
                <span><Monitor size={12} /> 右屏 · 外接显示器</span>
                <em>神经显微镜</em>
              </div>
              <div className="monitor">
                <div className="monitor-screen">
                  <div className="mock-topbar">
                    <span className="mock-brand"><i /> YOLO26 · 神经显微镜</span>
                    <span className="mock-live blue"><i /> 真实数据</span>
                  </div>
                  <div className="mock-neural">
                    <div className="mock-stations">
                      {STATIONS.map((station, index) => (
                        <span key={station.id} style={{ color: STATION_COLORS[index] }}>
                          <i style={{ background: STATION_COLORS[index] }} /> {station.name}
                        </span>
                      ))}
                    </div>
                    <div className="mock-neural-main">
                      <div className="mock-map">
                        {previewMap?.map ? (
                          <img src={`data:image/jpeg;base64,${previewMap.map}`} alt="特征图预览" />
                        ) : (
                          <div className="mock-map-grid" />
                        )}
                        <span className="mock-map-scan" />
                        <em>LAYER {String(previewMap?.i ?? "--").padStart(2, "0")}</em>
                      </div>
                      <div className="mock-slots">
                        <div className="mock-slots-title">300 查询槽位</div>
                        <div className="mock-slot-grid">
                          {Array.from({ length: 120 }).map((_, index) => {
                            const value = headData?.slots?.[index * 2] ?? 0;
                            return <i key={index} style={{ opacity: 0.12 + Math.min(0.88, value * 3) }} />;
                          })}
                        </div>
                        <div className="mock-top-row">
                          <span>激活</span>
                          <b>{headData?.active ?? 0}</b>
                        </div>
                      </div>
                    </div>
                    <div className="mock-rail">
                      {Array.from({ length: 24 }).map((_, index) => (
                        <span key={index} className={index === (previewMap?.i ?? 7) ? "on" : ""}>
                          {String(index).padStart(2, "0")}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>
                <div className="monitor-neck" />
                <div className="monitor-foot" />
              </div>
            </div>

            <div className="stage-flow">
              <span>同一路真实推理</span>
              <i />
              <span>WebSocket</span>
              <i />
              <span>两个独立窗口</span>
            </div>
          </motion.div>
        </section>

        <section className="story-grid">
          <motion.article className="story-card" variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0.05}>
            <div className="story-icon"><Eye size={19} /></div>
            <div className="story-index mono">01 · OBSERVE</div>
            <h3>先看见「确定程度」</h3>
            <p>不是一句空泛的 92%。人脸置信度 = 人物检测置信度 × 面部 5 个关键点的平均置信度，公式与数值都摊开给你看。</p>
          </motion.article>
          <motion.article className="story-card" variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0.12}>
            <div className="story-icon"><Layers3 size={19} /></div>
            <div className="story-index mono">02 · DISSECT</div>
            <h3>再把网络拆开</h3>
            <p>24 个顶层模块，23 层有真实特征图；另有 7 个卷积块显微镜，每次滑动展示 9 个真实乘积。所有画面来自同一次前向。</p>
          </motion.article>
          <motion.article className="story-card" variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0.19}>
            <div className="story-icon"><UserCheck size={19} /></div>
            <div className="story-index mono">03 · RECOGNIZE</div>
            <h3>最后认出「这是谁」</h3>
            <p>3 点人脸对齐 + EdgeFace-S 512 维特征。相似度超过阈值才算认识，注册一张脸，就能跨帧叫出名字。</p>
          </motion.article>
        </section>

        <section className="guide-panel">
          <div className="guide-copy">
            <div className="eyebrow">DUAL SCREEN SETUP</div>
            <h2>两块屏幕，各看各的</h2>
            <p>
              两个窗口是完全独立的页面，共享同一个后端推理线程。把右屏拖到外接显示器，各自按 F11，就能得到一块观察屏 + 一块神经显微镜。
            </p>
            <div className="guide-actions">
              <button className="btn btn-primary" onClick={openBoth}>
                <Users size={15} /> 打开两个窗口
              </button>
              <button className="btn btn-ghost" onClick={() => openWindow("/neural", "yolo26-neural", 1600, 940)}>
                <Sparkles size={14} /> 单独体验神经显微镜
              </button>
            </div>
          </div>
          <div className="guide-steps">
            <div className="guide-step">
              <span className="step-num mono">01</span>
              <div><b>打开双窗</b><p>允许浏览器弹出两个窗口，它们会自动连上同一路数据。</p></div>
            </div>
            <div className="guide-step">
              <span className="step-num mono">02</span>
              <div><b>把右屏拖走</b><p>把「神经显微镜」窗口拖到外接显示器，左屏留在笔记本。</p></div>
            </div>
            <div className="guide-step">
              <span className="step-num mono">03</span>
              <div><b>各自全屏</b><p>按 F11 或窗口按钮全屏。右屏可以用空格暂停导览，点击任意层自己探索。</p></div>
            </div>
          </div>
        </section>

        <section className="tech-strip">
          <span className="tech-label mono">POWERED BY</span>
          <span className="tech-item"><Cpu size={13} /> YOLO26n-Pose · {model?.params ? `${(model.params / 1e6).toFixed(1)}M` : "3.7M"} params</span>
          <span className="tech-item"><ScanFace size={13} /> EdgeFace-S · 512D</span>
          <span className="tech-item"><Layers3 size={13} /> {model?.cuda ? `CUDA ${model.cuda}` : "PyTorch"}</span>
          <span className="tech-item">
            <Network size={13} />{" "}
            {continuousProbe(model?.probe_interval, model?.probe_continuous)
              ? "连续探针刷新"
              : `${model?.probe_interval ?? 0.6}s 探针刷新`}
          </span>
          <span className="tech-note">全部推理在本机完成</span>
        </section>
      </main>

      <footer className="landing-footer">
        <span>YOLO26 · 神经显微镜</span>
        <span className="mono">LIVE FACE CONFIDENCE / REAL INTERMEDIATE MAPS / IDENTITY MATCHING</span>
        <span>{stats?.gpu_mem_mb ? `${stats.gpu_mem_mb.toFixed(0)} MB VRAM` : "LOCAL INFERENCE"}</span>
      </footer>

      <Toasts toasts={toasts} dismiss={dismiss} />
    </div>
  );
}
