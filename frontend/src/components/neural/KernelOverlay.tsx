import { Pause, Play } from "lucide-react";
import type { MicroData } from "../../types";

export function KernelOverlay({
  data,
  step,
  seq,
  playing,
  onTogglePlay,
}: {
  data: MicroData;
  step: number;
  seq: number | null;
  playing: boolean;
  onTogglePlay: () => void;
}) {
  const view = data.view;
  const steps = data.steps ?? [];
  if (!view?.image || steps.length === 0) return null;

  const index = Math.min(step, steps.length - 1);
  const current = steps[index];
  const [patchX, patchY, patchW, patchH] = view.patch;
  const windowLeft = patchX + (current.x / data.patch_size) * patchW;
  const windowTop = patchY + (current.y / data.patch_size) * patchH;
  const windowW = (data.kernel / data.patch_size) * patchW;
  const windowH = (data.kernel / data.patch_size) * patchH;

  return (
    <section
      className="micro-overlay"
      data-layer={data.layer}
      data-index={index}
      data-step={`${current.x}-${current.y}`}
      data-seq={seq ?? ""}
      aria-label="卷积取景窗"
    >
      <div className="mo-head">
        <span className="mo-kicker mono">CONV VIEWFINDER</span>
        <span className="mo-seq mono">{seq != null ? `同帧 #${seq}` : "实时"}</span>
      </div>
      <div className="mo-title">
        卷积正在读这里 <em className="mono">IN #{data.channel_in}</em>
      </div>
      <div className="mo-view">
        <img
          src={`data:image/jpeg;base64,${view.image}`}
          alt={`第 ${data.layer} 层卷积输入通道 ${data.channel_in}`}
          draggable={false}
        />
        <span
          className="mo-patch"
          style={{
            left: `${patchX * 100}%`,
            top: `${patchY * 100}%`,
            width: `${patchW * 100}%`,
            height: `${patchH * 100}%`,
          }}
        />
        <span
          className="mo-window"
          style={{
            left: `${windowLeft * 100}%`,
            top: `${windowTop * 100}%`,
            width: `${windowW * 100}%`,
            height: `${windowH * 100}%`,
          }}
        />
      </div>
      <div className="mo-foot mono">
        <span>
          第 {index + 1}/{steps.length} 次
        </span>
        <b>Σ {current.sum.toFixed(2)}</b>
        <button
          className="mo-play"
          onClick={onTogglePlay}
          title={playing ? "暂停同步滑动" : "继续同步滑动"}
        >
          {playing ? <Pause size={10} /> : <Play size={10} />}
        </button>
      </div>
      <p className="mo-note">底图是卷积输入，不是左侧这张输出层图</p>
    </section>
  );
}
