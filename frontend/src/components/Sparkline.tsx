import { useId } from "react";

export function Sparkline({
  values,
  width = 260,
  height = 56,
  color = "var(--cyan)",
  baseline = true,
}: {
  values: number[];
  width?: number;
  height?: number;
  color?: string;
  baseline?: boolean;
}) {
  const id = useId().replace(/:/g, "");
  const data = values.slice(-72);
  const padding = 4;
  const usable = Math.max(1, data.length - 1);
  const points = data.map((value, index) => {
    const x = padding + (index / usable) * (width - padding * 2);
    const y = height - padding - Math.min(1, Math.max(0, value)) * (height - padding * 2);
    return [x, y] as const;
  });
  const line = points.map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(1)},${y.toFixed(1)}`).join(" ");
  const area = points.length
    ? `${line} L${points[points.length - 1][0].toFixed(1)},${height - padding} L${points[0][0].toFixed(1)},${height - padding} Z`
    : "";
  const last = points[points.length - 1];

  return (
    <svg viewBox={`0 0 ${width} ${height}`} className="sparkline" preserveAspectRatio="none">
      <defs>
        <linearGradient id={`spark-${id}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.32" />
          <stop offset="100%" stopColor={color} stopOpacity="0" />
        </linearGradient>
      </defs>
      {baseline &&
        [0.25, 0.5, 0.75].map((fraction) => (
          <line
            key={fraction}
            x1={0}
            x2={width}
            y1={height * fraction}
            y2={height * fraction}
            stroke="var(--spark-baseline, rgba(126,160,200,.12))"
            strokeDasharray="2 5"
          />
        ))}
      {area && <path d={area} fill={`url(#spark-${id})`} />}
      {line && <path d={line} fill="none" stroke={color} strokeWidth={1.8} strokeLinejoin="round" />}
      {last && (
        <>
          <circle cx={last[0]} cy={last[1]} r={6} fill={color} opacity={0.16} />
          <circle cx={last[0]} cy={last[1]} r={2.4} fill={color} />
        </>
      )}
    </svg>
  );
}
