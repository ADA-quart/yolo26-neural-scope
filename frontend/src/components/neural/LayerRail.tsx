import { BrainCircuit } from "lucide-react";
import type { LayerMeta, MapEntry } from "../../types";
import { STATION_COLORS } from "../../types";

export function LayerRail({
  meta,
  maps,
  focus,
  onSelect,
  fresh,
}: {
  meta: LayerMeta[];
  maps: Record<number, MapEntry>;
  focus: number;
  onSelect: (index: number) => void;
  fresh: boolean;
}) {
  if (!meta.length) {
    return (
      <div className="layer-rail empty">
        {Array.from({ length: 24 }).map((_, index) => (
          <div className="rail-tile placeholder" key={index} />
        ))}
      </div>
    );
  }

  return (
    <div className={`layer-rail ${fresh ? "fresh" : ""}`}>
      {meta.map((layer) => {
        const entry = maps[layer.i];
        const active = layer.i === focus;
        const color = STATION_COLORS[layer.station] ?? STATION_COLORS[0];
        return (
          <button
            key={layer.i}
            className={`rail-tile ${active ? "active" : ""} ${layer.i === 23 ? "head-tile" : ""}`}
            style={{ ["--station" as string]: color }}
            onClick={() => onSelect(layer.i)}
            title={`${layer.name} · ${layer.type} · ${layer.shape?.slice(1).join("×") ?? ""} · ${layer.caption_zh}`}
          >
            <span className="rail-thumb">
              {layer.i === 23 ? (
                <BrainCircuit size={16} />
              ) : entry?.map ? (
                <img src={`data:image/jpeg;base64,${entry.map}`} alt={`层 ${layer.i}`} draggable={false} />
              ) : (
                <span className="rail-void" />
              )}
              <i className="rail-bar" style={{ width: `${(entry?.activity ?? 0.12) * 100}%` }} />
            </span>
            <span className="rail-foot">
              <span className="mono">{String(layer.i).padStart(2, "0")}</span>
              <em>{layer.type}</em>
            </span>
          </button>
        );
      })}
    </div>
  );
}
