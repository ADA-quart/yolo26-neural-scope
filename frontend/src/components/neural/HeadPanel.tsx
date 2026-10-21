import { BrainCircuit, Crosshair } from "lucide-react";
import type { FrameInfo, HeadData } from "../../types";

export function HeadPanel({
  head,
  frame,
  seq,
  full = false,
}: {
  head: HeadData | null;
  frame?: FrameInfo | null;
  seq?: number | null;
  full?: boolean;
}) {
  const slots = head?.slots ?? [];
  const maxScore = Math.max(0.0001, ...slots.slice(0, 300));
  const top = (head?.top ?? []).filter((item) => item.score > 0.05).slice(0, full ? 6 : 3);
  const active = head?.active ?? slots.filter((value) => value > 0.25).length;

  return (
    <div className={`head-panel ${full ? "head-panel-full" : ""}`} data-forward-seq={seq ?? ""}>
      <div className="head-stats">
        <div className="head-active">
          <span className="mono">亮格 · 正在输出</span>
          <b>{active}</b>
          <em>/ {head?.queries ?? 300}</em>
        </div>
        <div className="head-stats-right mono">
          <span>端到端 · 无需 NMS</span>
          <span>每格 = 1 个查询槽位</span>
        </div>
      </div>

      {full ? (
        <div className="head-full-body">
          <div
            className="head-canvas"
            style={{ ["--ar" as string]: frame ? frame.w / frame.h : 16 / 9 }}
          >
            {frame?.jpeg ? (
              <img src={`data:image/jpeg;base64,${frame.jpeg}`} alt="决策头输入画面" draggable={false} />
            ) : (
              <div className="head-canvas-empty">等待实时画面</div>
            )}
            {top.map((item, index) => {
              const [x1, y1, x2, y2] = item.bbox;
              return (
                <span
                  key={item.slot}
                  className={`proposal-box ${index === 0 ? "top" : ""}`}
                  style={{
                    left: `${x1 * 100}%`,
                    top: `${y1 * 100}%`,
                    width: `${(x2 - x1) * 100}%`,
                    height: `${(y2 - y1) * 100}%`,
                  }}
                >
                  <em className="mono">
                    Q{item.slot} · {(item.score * 100).toFixed(1)}%
                  </em>
                </span>
              );
            })}
            <div className="head-canvas-label mono">
              <Crosshair size={10} /> 每个框 = 一个查询槽位的直接输出
            </div>
          </div>
          <div className="head-full-side">
            <div className="head-full-title">
              <BrainCircuit size={15} /> 300 个查询槽位，各给一个答案
            </div>
            <p>
              每个槽位直接输出框、置信度、类别和 17 个关键点；一对一匹配让每个目标只保留一个框，
              因此不需要 NMS。分数低于阈值的槽位不会进入最终结果。
            </p>
            <div className="top-list">
              {top.length === 0 && <div className="empty-state">当前帧没有高分槽位</div>}
              {top.map((item) => (
                <div className="top-item" key={item.slot}>
                  <span className="mono">Q{String(item.slot).padStart(3, "0")}</span>
                  <div className="top-bar"><i style={{ width: `${Math.min(1, item.score) * 100}%` }} /></div>
                  <b className="mono">{(item.score * 100).toFixed(1)}%</b>
                </div>
              ))}
            </div>
            <div className="head-live-note">
              <span className="live-dot is-live" /> 同一次前向里，300 个槽位一起更新
            </div>
          </div>
        </div>
      ) : (
        <>
          <div className="slot-grid-compact">
            {Array.from({ length: 300 }).map((_, index) => {
              const score = slots[index] ?? 0;
              const intensity = score / maxScore;
              const hot = score > 0.25;
              return (
                <i
                  key={index}
                  className={hot ? "hot" : ""}
                  style={{ opacity: 0.08 + Math.min(0.92, intensity * 1.15) }}
                  title={`槽位 ${index} · ${score.toFixed(3)}`}
                />
              );
            })}
          </div>
          <div className="top-list compact">
            {top.length === 0 && <div className="head-empty">等待高分槽位…</div>}
            {top.map((item) => (
              <div className="top-item" key={item.slot}>
                <span className="mono">Q{String(item.slot).padStart(3, "0")}</span>
                <div className="top-bar"><i style={{ width: `${Math.min(1, item.score) * 100}%` }} /></div>
                <b className="mono">{(item.score * 100).toFixed(1)}%</b>
              </div>
            ))}
          </div>
        </>
      )}

      {full && (
        <div className="head-slot-grid">
          {Array.from({ length: 300 }).map((_, index) => {
            const score = slots[index] ?? 0;
            const hot = score > 0.25;
            return (
              <button
                key={index}
                className={`head-slot ${hot ? "hot" : ""}`}
                title={`查询槽位 ${index} · 置信度 ${score.toFixed(3)}`}
              >
                <i style={{ height: `${Math.max(4, (score / maxScore) * 100)}%` }} />
                <span className="mono">{index}</span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
