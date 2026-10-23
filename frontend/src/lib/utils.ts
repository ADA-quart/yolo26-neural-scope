export function pct(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return `${(value * 100).toFixed(digits)}%`;
}

export function num(value: number | null | undefined, digits = 0): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "--";
  return value.toLocaleString("en-US", { maximumFractionDigits: digits });
}

export function compact(value: number | null | undefined): string {
  if (value === null || value === undefined) return "--";
  if (value >= 1_000_000) return `${(value / 1_000_000).toFixed(2)}M`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}K`;
  return `${value}`;
}

export function shapeText(shape: number[] | null | undefined): string {
  if (!shape || shape.length === 0) return "--";
  return shape.join(" × ");
}

export function featureShape(shape: number[] | null | undefined): string {
  if (!shape) return "--";
  if (shape.length === 3) return `${shape[0]} × ${shape[1]} × ${shape[2]}`;
  if (shape.length === 2) return `${shape[0]} × ${shape[1]}`;
  return shape.join(" × ");
}

export function rateLabel(value: number): string {
  if (value >= 0.85) return "非常确定";
  if (value >= 0.65) return "比较确定";
  if (value >= 0.4) return "仍在观察";
  return "信号很弱";
}

export function simLabel(value: number | null | undefined): string {
  if (value === null || value === undefined) return "等待比对";
  if (value >= 0.65) return "高度相似";
  if (value >= 0.45) return "相似";
  if (value >= 0.3) return "有些像";
  return "不相似";
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function continuousProbe(interval?: number | null, continuous?: boolean): boolean {
  if (continuous !== undefined) return continuous;
  return interval !== undefined && interval !== null && interval <= 0;
}

export function probeCadence(interval?: number | null, continuous?: boolean): string {
  if (continuousProbe(interval, continuous)) return "探针连续";
  return `探针每 ${(interval ?? 0.6).toFixed(1)}s`;
}

export async function api<T = unknown>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: init?.body instanceof FormData ? undefined : { "Content-Type": "application/json", ...init?.headers },
  });
  let payload: unknown = null;
  try {
    payload = await response.json();
  } catch {
    // keep null
  }
  if (!response.ok) {
    const message =
      payload && typeof payload === "object" && "error" in payload
        ? String((payload as { error: unknown }).error)
        : `请求失败 (${response.status})`;
    throw new Error(message);
  }
  return payload as T;
}

export function openWindow(path: string, name: string, width: number, height: number): Window | null {
  const win = window.open(
    path,
    name,
    `popup=yes,width=${width},height=${height},left=40,top=40`,
  );
  win?.focus();
  return win;
}
