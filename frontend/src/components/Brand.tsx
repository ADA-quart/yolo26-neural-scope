import { motion } from "framer-motion";

export function Brand({
  subtitle,
  compact = false,
}: {
  subtitle?: string;
  compact?: boolean;
}) {
  return (
    <div className={`brand ${compact ? "brand-compact" : ""}`}>
      <div className="brand-mark" aria-hidden>
        <svg viewBox="0 0 44 44" fill="none">
          <circle cx="22" cy="22" r="19" stroke="url(#brandGrad)" strokeWidth="1.2" opacity="0.55" />
          <circle cx="22" cy="22" r="12.5" stroke="url(#brandGrad)" strokeWidth="1.5" />
          <path d="M4 22c5.4-7.2 11.4-10.8 18-10.8S34.6 14.8 40 22c-5.4 7.2-11.4 10.8-18 10.8S9.4 29.2 4 22Z" stroke="url(#brandGrad)" strokeWidth="1.6" />
          <circle cx="22" cy="22" r="4.4" fill="var(--cyan)" />
          <circle cx="22" cy="22" r="7.6" stroke="var(--cyan)" strokeOpacity="0.3">
            <animate attributeName="r" values="7.6;10.5;7.6" dur="2.8s" repeatCount="indefinite" />
            <animate attributeName="stroke-opacity" values="0.32;0;0.32" dur="2.8s" repeatCount="indefinite" />
          </circle>
          <defs>
            <linearGradient id="brandGrad" x1="4" y1="6" x2="40" y2="38" gradientUnits="userSpaceOnUse">
              <stop stopColor="var(--cyan)" />
              <stop offset="0.55" stopColor="var(--blue)" />
              <stop offset="1" stopColor="var(--violet)" />
            </linearGradient>
          </defs>
        </svg>
      </div>
      <div className="brand-copy">
        <div className="brand-title">
          YOLO<span>26</span>
          <em>神经显微镜</em>
        </div>
        {subtitle && <div className="brand-sub">{subtitle}</div>}
      </div>
    </div>
  );
}

export function LiveDot({ connected }: { connected: boolean }) {
  return (
    <motion.span
      className={`live-dot ${connected ? "is-live" : "is-down"}`}
      animate={connected ? { opacity: [1, 0.35, 1] } : { opacity: 1 }}
      transition={{ duration: 2.2, repeat: Infinity, ease: "easeInOut" }}
    />
  );
}

export function ConnectionPill({ connected, label }: { connected: boolean; label?: string }) {
  return (
    <span className={`conn-pill ${connected ? "on" : "off"}`}>
      <LiveDot connected={connected} />
      {label ?? (connected ? "数据流在线" : "正在重连")}
    </span>
  );
}
