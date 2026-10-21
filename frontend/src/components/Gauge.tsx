import { useId } from "react";
import { rateLabel } from "../lib/utils";

function colorFor(value: number): string {
  if (value >= 0.82) return "var(--green)";
  if (value >= 0.62) return "var(--cyan)";
  if (value >= 0.4) return "var(--amber)";
  return "var(--red)";
}

export function Gauge({
  value,
  size = 186,
  caption,
}: {
  value: number;
  size?: number;
  caption?: string;
}) {
  const id = useId().replace(/:/g, "");
  const clamped = Math.min(1, Math.max(0, value));
  const radius = size * 0.39;
  const circumference = 2 * Math.PI * radius;
  const color = colorFor(clamped);
  const ticks = 48;

  return (
    <div className="gauge" style={{ width: size, height: size }}>
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size}>
        <defs>
          <linearGradient id={`gauge-${id}`} x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.55" />
            <stop offset="100%" stopColor={color} />
          </linearGradient>
          <filter id={`glow-${id}`} x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur stdDeviation="4" result="blur" />
            <feMerge>
              <feMergeNode in="blur" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        {Array.from({ length: ticks }).map((_, index) => {
          const angle = (index / ticks) * Math.PI * 2 - Math.PI / 2;
          const active = index / ticks <= clamped;
          const inner = radius + size * 0.075;
          const outer = radius + size * 0.1;
          return (
            <line
              key={index}
              x1={size / 2 + Math.cos(angle) * inner}
              y1={size / 2 + Math.sin(angle) * inner}
              x2={size / 2 + Math.cos(angle) * outer}
              y2={size / 2 + Math.sin(angle) * outer}
              stroke={active ? color : "var(--gauge-track, rgba(126,160,200,.18))"}
              strokeWidth={active ? 1.7 : 1}
              strokeLinecap="round"
            />
          );
        })}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--gauge-track-soft, rgba(126,160,200,.12))"
          strokeWidth={size * 0.045}
        />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={`url(#gauge-${id})`}
          strokeWidth={size * 0.045}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={circumference * (1 - clamped)}
          transform={`rotate(-90 ${size / 2} ${size / 2})`}
          filter={`url(#glow-${id})`}
          style={{ transition: "stroke-dashoffset .55s cubic-bezier(.22,.8,.3,1), stroke .4s" }}
        />
      </svg>
      <div className="gauge-center">
        <div className="gauge-value" style={{ color }}>
          {(clamped * 100).toFixed(1)}
          <span>%</span>
        </div>
        <div className="gauge-label">{rateLabel(clamped)}</div>
        {caption && <div className="gauge-caption">{caption}</div>}
      </div>
    </div>
  );
}
