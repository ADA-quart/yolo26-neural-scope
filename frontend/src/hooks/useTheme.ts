import { useCallback, useEffect, useState } from "react";

export type Theme = "night" | "daylight";

const STORAGE_KEY = "yolo26-theme";

function readTheme(): Theme {
  return document.documentElement.dataset.theme === "daylight" ? "daylight" : "night";
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(readTheme);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    const meta = document.querySelector('meta[name="theme-color"]');
    meta?.setAttribute("content", theme === "daylight" ? "#eef1f5" : "#05070f");
    try {
      window.localStorage.setItem(STORAGE_KEY, theme);
    } catch {
      // Private-mode storage can be unavailable; the in-memory theme still works.
    }
  }, [theme]);

  useEffect(() => {
    const sync = (event: StorageEvent) => {
      if (event.key !== STORAGE_KEY) return;
      setTheme(event.newValue === "daylight" ? "daylight" : "night");
    };
    window.addEventListener("storage", sync);
    return () => window.removeEventListener("storage", sync);
  }, []);

  const toggle = useCallback(
    () => setTheme((value) => (value === "daylight" ? "night" : "daylight")),
    [],
  );

  return { theme, toggle };
}
