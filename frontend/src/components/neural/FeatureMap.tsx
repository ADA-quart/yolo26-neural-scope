import { ScanLine } from "lucide-react";
import { memo, type CSSProperties } from "react";
import { pct } from "../../lib/utils";
import type { MapEntry } from "../../types";

export const FeatureMap = memo(function FeatureMap({
  entry,
  badge,
  className = "",
  tint = "cyan",
  showScan = true,
  style,
}: {
  entry: MapEntry | null | undefined;
  badge?: string;
  className?: string;
  tint?: "cyan" | "blue" | "violet" | "amber";
  showScan?: boolean;
  style?: CSSProperties;
}) {
  return (
    <div className={`feature-map tint-${tint} ${className}`} style={style}>
      {entry?.map ? (
        <img src={`data:image/jpeg;base64,${entry.map}`} alt={`第 ${entry.i} 层特征图`} draggable={false} />
      ) : (
        <div className="fm-fallback">
          <div className="fm-grid" />
          <span>等待探针数据</span>
        </div>
      )}
      <div className="fm-colorwash" />
      <div className="fm-mesh" />
      {showScan && <span className="fm-sweep" />}
      {badge && <span className="fm-badge">{badge}</span>}
      {entry?.activity != null && (
        <span className="fm-activity mono">
          <ScanLine size={10} /> 活度 {pct(entry.activity, 1)}
        </span>
      )}
      {entry?.shape && <span className="fm-shape mono">{entry.shape.slice(1).join("×")}</span>}
    </div>
  );
});
