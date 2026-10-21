import { Boxes, Grid3X3, Layers, MoveUpRight, Network, ScanEye } from "lucide-react";
import { STATION_COLORS } from "../../types";
import type { LayerMeta } from "../../types";

const ICONS: Record<string, typeof Layers> = {
  Conv: Grid3X3,
  C3k2: Network,
  SPPF: Layers,
  C2PSA: ScanEye,
  Upsample: MoveUpRight,
  Concat: Boxes,
  Pose26: Network,
};

const FLOW_LABELS: Record<string, string> = {
  Conv: "乘加 + 激活",
  C3k2: "分流后合并",
  SPPF: "多尺度池化",
  C2PSA: "注意力重分配",
  Upsample: "空间放大 ×2",
  Concat: "通道拼接",
  Pose26: "300 槽位输出",
};

const FORMULAS: Record<string, string> = {
  Conv: "y = 激活( Σ w · x + b )",
  C3k2: "y = Concat( C3k(x), C3k(x), x )",
  SPPF: "y = Concat( MaxPool5, MaxPool9, MaxPool13, x )",
  C2PSA: "y = x + 注意力( x )",
  Upsample: "H × W → 2H × 2W",
  Concat: "C_out = C₁ + C₂ + …",
  Pose26: "300 × ( 框 + 分数 + 类别 + 17 关键点 )",
};

export function LayerStructure({ layer, prevShape }: { layer: LayerMeta; prevShape?: number[] | null }) {
  const type = layer.type;
  const shapeLabel = layer.shape?.slice(1).join("×") ?? "--";
  const prevLabel = prevShape?.slice(1).join("×") ?? "--";
  return (
    <section className="structure-card panel">
      <div className="micro-head">
        <div>
          <div className="panel-kicker">STRUCTURE · 结构示意</div>
          <div className="micro-title">
            {(() => {
              const Icon = ICONS[type] ?? Layers;
              return <Icon size={14} />;
            })()}
            {type} 层在做什么
          </div>
        </div>
        <span className="chip">静态示意</span>
      </div>

      <div className={`structure-diagram type-${type}`}>
        <span className="diagram-formula mono">{FORMULAS[type] ?? "逐层变换"}</span>
        {type === "Conv" && (
          <div className="diag-conv">
            <div className="diag-grid">
              {Array.from({ length: 36 }).map((_, index) => <i key={index} />)}
              <span className="diag-kernel" />
            </div>
            <span className="diag-arrow">→</span>
            <div className="diag-grid small">
              {Array.from({ length: 16 }).map((_, index) => <i key={index} />)}
            </div>
          </div>
        )}
        {type === "C3k2" && (
          <div className="diag-branches">
            <span className="diag-input">IN</span>
            <div className="diag-branch-lines">
              <i /><i /><i />
            </div>
            <div className="diag-branch-boxes">
              <span><em />C3k</span>
              <span><em />C3k</span>
              <span className="short">捷径</span>
            </div>
            <span className="diag-arrow">→</span>
            <span className="diag-output">OUT</span>
          </div>
        )}
        {type === "SPPF" && (
          <div className="diag-sppf">
            <span className="sppf-box b1">5×5</span>
            <span className="sppf-box b2">9×9</span>
            <span className="sppf-box b3">13×13</span>
            <span className="sppf-note">不同大小的池化窗口串联，一次看到多种尺度</span>
          </div>
        )}
        {type === "C2PSA" && (
          <div className="diag-attention">
            {Array.from({ length: 48 }).map((_, index) => (
              <i key={index} style={{ opacity: 0.1 + ((index * 37) % 10) / 12 }} />
            ))}
            <span className="attention-core" />
            <em>位置敏感注意力</em>
          </div>
        )}
        {type === "Upsample" && (
          <div className="diag-upsample">
            <div className="diag-grid mini">
              {Array.from({ length: 4 }).map((_, index) => <i key={index} />)}
            </div>
            <span className="diag-arrow">→</span>
            <div className="diag-grid mini large">
              {Array.from({ length: 16 }).map((_, index) => <i key={index} />)}
            </div>
          </div>
        )}
        {type === "Concat" && (
          <div className="diag-concat">
            <span className="concat-stack"><i /><i /><i /></span>
            <span className="diag-arrow">+</span>
            <span className="concat-stack alt"><i /><i /></span>
            <span className="diag-arrow">→</span>
            <span className="concat-stack merged"><i /><i /><i /><i /><i /></span>
          </div>
        )}
        {(type === "Pose26" || type === "Detect") && (
          <div className="diag-head">
            <div className="diag-queries">
              {Array.from({ length: 60 }).map((_, index) => (
                <i key={index} style={{ opacity: index % 13 === 0 ? 1 : 0.14 }} />
              ))}
            </div>
            <span className="diag-arrow">→</span>
            <div className="diag-decision">
              <span>框</span><span>置信度</span><span>关键点</span>
            </div>
          </div>
        )}
      </div>

      <div className="structure-flow">
        <span className="flow-node">
          <em>IN</em>
          <b className="mono">{prevLabel}</b>
        </span>
        <span className="flow-link"><i /></span>
        <span className="flow-node accent" style={{ ["--station" as string]: STATION_COLORS[layer.station] }}>
          <em>{type}</em>
          <b>{FLOW_LABELS[type] ?? "逐层变换"}</b>
        </span>
        <span className="flow-link"><i /></span>
        <span className="flow-node">
          <em>OUT</em>
          <b className="mono">{shapeLabel}</b>
        </span>
      </div>

      <p className="structure-caption">
        {layer.detail_zh}
        <em>示意动画用于解释结构，不是当前帧的数据。</em>
      </p>

      <div className="structure-props">
        <div><span>类型</span><b>{layer.type}</b></div>
        <div><span>输出形状</span><b className="mono">{layer.shape?.slice(1).join(" × ") ?? "--"}</b></div>
        <div><span>参数量</span><b className="mono">{layer.params.toLocaleString("en-US")}</b></div>
        <div><span>GPU 耗时</span><b className="mono">{layer.gpu_ms != null ? `${layer.gpu_ms.toFixed(2)} ms` : "--"}</b></div>
      </div>
    </section>
  );
}
