import { Radio } from "lucide-react";
import { STATIONS, STATION_COLORS } from "../../types";

export function StationBar({
  active,
  focus,
  onPick,
  live,
}: {
  active: number;
  focus: number;
  onPick: (station: number) => void;
  live: boolean;
}) {
  return (
    <div className="station-bar">
      <div className="station-track">
        {STATIONS.map((station, index) => (
          <button
            key={station.id}
            className={`station-item ${index === active ? "active" : ""} ${index < active ? "done" : ""}`}
            onClick={() => onPick(index)}
            style={{ ["--station" as string]: STATION_COLORS[index] }}
          >
            <span className="station-node">
              <i />
              {index < 3 && <em />}
            </span>
            <span className="station-copy">
              <span className="station-name">
                {station.name}
                <em className="mono">{station.en}</em>
              </span>
              <span className="station-range mono">LAYER {station.range}</span>
            </span>
          </button>
        ))}
        <span className="station-live">
          <Radio size={11} />
          {live ? "探针已接入" : "等待探针"}
        </span>
      </div>
      <div className="station-hint">
        <span className="mono">NOW · 层 {String(focus).padStart(2, "0")}</span>
        <p>{STATIONS[active]?.hint}</p>
      </div>
    </div>
  );
}
