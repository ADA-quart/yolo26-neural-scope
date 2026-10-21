import { Moon, Sun } from "lucide-react";
import { useTheme } from "../hooks/useTheme";

export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const daylight = theme === "daylight";
  const label = daylight ? "夜间模式" : "日光模式";

  return (
    <button
      className={`btn btn-ghost theme-toggle ${daylight ? "is-daylight" : ""}`}
      onClick={toggle}
      title={`${label}（强光下更清楚，两个窗口同步）`}
      aria-label={`切换到${label}`}
      aria-pressed={daylight}
    >
      {daylight ? <Moon size={15} /> : <Sun size={15} />}
      <span>{daylight ? "夜间" : "日光"}</span>
    </button>
  );
}
