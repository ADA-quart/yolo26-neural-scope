import { motion } from "framer-motion";
import { Grid3X3, Pause, Play, Sigma } from "lucide-react";
import { useMemo } from "react";
import type { LayerMeta, MicroData } from "../../types";

function valueColor(value: number, maxAbs: number): string {
  const t = Math.min(1, Math.abs(value) / (maxAbs || 1));
  const strength = Math.round(8 + t * 72);
  return `color-mix(in srgb, var(${value >= 0 ? "--heat-pos" : "--heat-neg"}) ${strength}%, transparent)`;
}

function fmt(value: number): string {
  if (Math.abs(value) >= 10) return value.toFixed(1);
  return value.toFixed(2).replace(/0$/, "");
}

export function ConvMicroscope({
  data,
  layer,
  engineer,
  step,
  playing,
  locked,
  onStep,
  onTogglePlay,
  onHover,
}: {
  data: MicroData;
  layer: LayerMeta | undefined;
  engineer: boolean;
  step: number;
  playing: boolean;
  locked: boolean;
  onStep: (index: number) => void;
  onTogglePlay: () => void;
  onHover: (locked: boolean) => void;
}) {
  const steps = data.steps ?? [];
  const current = steps[Math.min(step, steps.length - 1)] ?? null;
  const patchMax = useMemo(() => Math.max(...data.patch.flat().map(Math.abs), 0.001), [data.patch]);
  const weightMax = useMemo(() => Math.max(...data.weights.flat().map(Math.abs), 0.001), [data.weights]);
  const productMax = useMemo(
    () => Math.max(...(current?.products ?? [0]).map(Math.abs), 0.001),
    [current],
  );

  const kernelPosition = current
    ? {
        left: `${(current.x / data.patch_size) * 100}%`,
        top: `${(current.y / data.patch_size) * 100}%`,
        width: `${(data.kernel / data.patch_size) * 100}%`,
        height: `${(data.kernel / data.patch_size) * 100}%`,
      }
    : undefined;

  return (
    <section className="micro-card panel">
      <div className="micro-head">
        <div>
          <div className="panel-kicker">CONV MICROSCOPE · 真实卷积</div>
          <div className="micro-title">
            <Grid3X3 size={14} /> {data.kernel}×{data.kernel} 卷积核正在滑过输入
          </div>
        </div>
        <button
          className="micro-play"
          onClick={onTogglePlay}
          title={playing ? "暂停乘加演示" : "继续乘加演示"}
        >
          {playing ? <Pause size={12} /> : <Play size={12} />}
          {playing ? "演示中" : "已暂停"}
        </button>
      </div>

      <div className="micro-meta mono">
        <span>IN #{data.channel_in} / {data.channels_in} CH</span>
        <span>OUT #{data.channel_out} / {data.channels_out} CH</span>
        <span>{data.spatial[0]}×{data.spatial[1]} MAP</span>
        <span>STRIDE {data.stride}</span>
      </div>

      <div className="micro-arrays">
        <div className="array-block">
          <div className="array-label">
            <span>输入切片</span>
            <em className="mono">{data.patch_size}×{data.patch_size}</em>
          </div>
          <div
            className="patch-grid"
            style={{ gridTemplateColumns: `repeat(${data.patch_size}, minmax(0, 1fr))` }}
            onMouseEnter={() => onHover(true)}
            onMouseLeave={() => onHover(false)}
          >
            {data.patch.map((row, y) =>
              row.map((value, x) => {
                const inside =
                  current &&
                  x >= current.x &&
                  x < current.x + data.kernel &&
                  y >= current.y &&
                  y < current.y + data.kernel;
                return (
                  <span
                    key={`${x}-${y}`}
                    className={`value-cell ${inside ? "inside" : ""}`}
                    style={{ background: valueColor(value, patchMax) }}
                  >
                    {fmt(value)}
                  </span>
                );
              }),
            )}
            {current && <span className="kernel-window" style={kernelPosition} />}
          </div>
        </div>

        <div className="micro-op mono">×</div>

        <div className="array-block kernel-block">
          <div className="array-label">
            <span>卷积核</span>
            <em className="mono">真实权重</em>
          </div>
          <div className="kernel-grid" style={{ gridTemplateColumns: `repeat(${data.kernel}, minmax(0, 1fr))` }}>
            {data.weights.map((row, y) =>
              row.map((value, x) => (
                <span key={`${x}-${y}`} className="value-cell" style={{ background: valueColor(value, weightMax) }}>
                  {fmt(value)}
                </span>
              )),
            )}
          </div>
          <div className="kernel-tag mono">层 {String(data.layer).padStart(2, "0")} · {layer?.type ?? "Conv"}</div>
        </div>

        <div className="micro-op mono">=</div>

        <div className="array-block products-block">
          <div className="array-label">
            <span>逐项乘积</span>
            <em className="mono">9 项</em>
          </div>
          <div className="kernel-grid" style={{ gridTemplateColumns: `repeat(${data.kernel}, minmax(0, 1fr))` }}>
            {(current?.products ?? []).map((value, index) => (
              <span key={index} className="value-cell product" style={{ background: valueColor(value, productMax) }}>
                {fmt(value)}
              </span>
            ))}
          </div>
          <div className="kernel-tag mono">第 {Math.min(step + 1, steps.length)} / {steps.length} 次滑动</div>
        </div>
      </div>

      <div className="micro-sum">
        <div className="sum-left">
          <Sigma size={15} />
          <span>9 个乘积相加 = 卷积和（还没加偏置、归一化和激活）</span>
        </div>
        <motion.b
          key={current?.sum}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          className="mono"
        >
          {current ? current.sum.toFixed(2) : "--"}
        </motion.b>
      </div>

      <div className="micro-steps">
        {steps.map((item, index) => (
          <button
            key={`${item.x}-${item.y}`}
            className={`step-pill ${index === Math.min(step, steps.length - 1) ? "active" : ""}`}
            onClick={() => onStep(index)}
            title={`窗口位置 (${item.x}, ${item.y}) · 和 ${item.sum.toFixed(2)}`}
          >
            <span className="mono">{String(index + 1).padStart(2, "0")}</span>
            <i />
          </button>
        ))}
      </div>

      <p className="micro-note">
        {engineer
          ? `输入通道按 ROI 方差自动选取，输出通道按 ROI 内的响应强度选取。窗口按 stride=${data.stride} 移动，${steps.length} 个输出位置共用同一组 ${data.kernel}×${data.kernel} 权重；左侧取景窗用同一通道的更大上下文同步显示窗口位置。这里只演示乘加和，真实模块还会接偏置、BN 和 SiLU。`
          : "把输入的一小块和卷积核叠在一起，对应位置相乘，再把 9 个结果加起来。卷积核滑到下一个位置，重复同一件事——这就是卷积。左侧取景窗会跟着同步滑动；图中只算到乘加和，后面的偏置、归一化和激活没有画出来。"}
      </p>
    </section>
  );
}
