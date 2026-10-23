export type LayerMeta = {
  i: number;
  name: string;
  type: string;
  group: string;
  station: number;
  params: number;
  shape: number[] | null;
  kernel: number | null;
  stride: number | null;
  padding: number | null;
  groups: number | null;
  is_conv: boolean;
  caption_zh: string;
  detail_zh: string;
  gpu_ms: number | null;
};

export type MapEntry = {
  i: number;
  map: string | null;
  activity?: number;
  shape?: number[];
};

export type MicroStep = {
  x: number;
  y: number;
  products: number[];
  sum: number;
};

export type MicroView = {
  image: string;
  patch: [number, number, number, number];
  context: number;
  source: [number, number];
};

export type MicroData = {
  layer: number;
  patch: number[][];
  weights: number[][];
  steps: MicroStep[];
  view?: MicroView | null;
  kernel: number;
  stride: number;
  patch_size: number;
  channel_in: number;
  channel_out: number;
  channels_in: number;
  channels_out: number;
  spatial: [number, number];
};

export type HeadTop = {
  slot: number;
  score: number;
  bbox: [number, number, number, number];
};

export type HeadData = {
  slots: number[];
  top: HeadTop[];
  active: number;
  queries: number;
};

export type FaceIdentity = {
  status: "pending" | "known" | "unknown";
  known: boolean;
  name: string;
  similarity: number | null;
  best_name: string | null;
};

export type FaceInfo = {
  id: number;
  box: [number, number, number, number];
  keypoints: [number, number, number][];
  person_conf: number;
  face_conf: number;
  kp_conf: number;
  eye_conf: number;
  identity: FaceIdentity;
  crop: string;
  selected: boolean;
};

export type FrameInfo = {
  w: number;
  h: number;
  jpeg: string;
};

export type SourceInfo = {
  kind: "demo" | "file" | "webcam" | "stream";
  value: string;
  label: string;
  loop: boolean;
  start: number;
  end: number;
};

export type BestInfo = {
  seq: number;
  face_conf: number;
  face_id: number;
  t: number;
};

export type HoldInfo = BestInfo & { active: true };

export type Stats = {
  fps: number;
  infer_ms: number;
  probe_ms: number;
  faces: number;
  frame: number;
  gpu_mem_mb?: number;
  probe_age?: number | null;
  best?: BestInfo | null;
  held?: boolean;
  runtime?: "tensorrt" | "pytorch";
};

export type GalleryItem = {
  id: string;
  name: string;
  demo: boolean;
  created: number;
  thumb: string;
};

export type EngineState = {
  paused: boolean;
  source: SourceInfo;
  selected: number | null;
  gallery: GalleryItem[];
  stats: Stats;
  hold?: HoldInfo | null;
  error: string;
};

export type ModelInfo = {
  name: string;
  params: number;
  layers: number;
  input: number;
  device: string;
  gpu: string;
  cuda: string;
  torch: string;
  weights: string;
  embedder: string;
  match_threshold: number;
  probe_interval: number;
  runtime?: string;
  engine?: string | null;
  meta: LayerMeta[];
};

export type LiveMsg = {
  type: "live";
  seq: number;
  t: number;
  frame: FrameInfo;
  faces: FaceInfo[];
  selected: number | null;
  source: SourceInfo;
  held?: boolean;
  hold?: BestInfo | null;
  stats: {
    infer_ms: number;
    probe_ms: number | null;
    fps: number;
    probed: boolean;
    runtime?: "tensorrt" | "pytorch";
  };
};

export type ProbeMsg = {
  type: "probe";
  seq: number;
  t: number;
  maps: MapEntry[];
  micro: Record<string, MicroData>;
  head: HeadData | null;
  roi: [number, number, number, number] | null;
  frame: FrameInfo;
  faces: FaceInfo[];
  face: { id: number; face_conf: number; identity: FaceIdentity } | null;
  took_ms: number;
  held?: boolean;
  hold?: BestInfo | null;
};

export type HeadMsg = {
  type: "head";
  seq: number;
  t: number;
  head: HeadData;
  selected: number | null;
};

export type StatusMsg = { type: "status" } & Stats;

export type StateMsg = { type: "state" } & EngineState;

export type HelloMsg = {
  type: "hello";
  state: EngineState;
  model: ModelInfo;
  live: LiveMsg | null;
  probe: ProbeMsg | null;
};

export type ToastMsg = { type: "toast"; message: string };
export type ErrorMsg = { type: "error"; message: string };

export type FeedMessage =
  | HelloMsg
  | LiveMsg
  | ProbeMsg
  | HeadMsg
  | StatusMsg
  | StateMsg
  | ToastMsg
  | ErrorMsg;

export type Toast = {
  id: number;
  message: string;
  tone: "info" | "warn" | "error" | "success";
};

export const STATIONS = [
  { id: 0, name: "找边缘", en: "EDGES", range: "0–2", hint: "浅层只看得见局部反差：边缘、色块、纹理。" },
  { id: 1, name: "认形状", en: "SHAPES", range: "3–10", hint: "把边缘组合成眼睛、鼻子、轮廓这些局部形状。" },
  { id: 2, name: "汇合尺度", en: "FUSION", range: "11–22", hint: "把细节和语义拼在一起，既看得清又看得懂。" },
  { id: 3, name: "下结论", en: "DECIDE", range: "23", hint: "300 个查询槽位同时给出框、置信度和关键点。" },
] as const;

export const STATION_COLORS = ["var(--cyan)", "var(--blue)", "var(--violet)", "var(--amber)"];
