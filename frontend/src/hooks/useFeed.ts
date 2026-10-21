import { useCallback, useEffect, useRef, useState } from "react";
import type {
  EngineState,
  HeadMsg,
  LiveMsg,
  ModelInfo,
  ProbeMsg,
  Stats,
  Toast,
  FeedMessage,
} from "../types";

type Feed = {
  connected: boolean;
  live: LiveMsg | null;
  probe: ProbeMsg | null;
  head: HeadMsg | null;
  state: EngineState | null;
  model: ModelInfo | null;
  stats: Stats | null;
  toasts: Toast[];
  notify: (message: string, tone?: Toast["tone"]) => void;
  dismiss: (id: number) => void;
};

let toastId = 0;

export function useFeed(): Feed {
  const [connected, setConnected] = useState(false);
  const [live, setLive] = useState<LiveMsg | null>(null);
  const [probe, setProbe] = useState<ProbeMsg | null>(null);
  const [head, setHead] = useState<HeadMsg | null>(null);
  const [state, setState] = useState<EngineState | null>(null);
  const [model, setModel] = useState<ModelInfo | null>(null);
  const [stats, setStats] = useState<Stats | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const timers = useRef<Map<number, number>>(new Map());

  const dismiss = useCallback((id: number) => {
    setToasts((items) => items.filter((item) => item.id !== id));
    const timer = timers.current.get(id);
    if (timer) {
      window.clearTimeout(timer);
      timers.current.delete(id);
    }
  }, []);

  const notify = useCallback(
    (message: string, tone: Toast["tone"] = "info") => {
      const id = ++toastId;
      setToasts((items) => [...items.slice(-3), { id, message, tone }]);
      const timer = window.setTimeout(() => {
        setToasts((items) => items.filter((item) => item.id !== id));
        timers.current.delete(id);
      }, tone === "error" ? 5200 : 3200);
      timers.current.set(id, timer);
    },
    [],
  );

  useEffect(() => {
    let socket: WebSocket | null = null;
    let retry = 0;
    let closed = false;
    let reconnectTimer = 0;

    const connect = () => {
      if (closed) return;
      const protocol = window.location.protocol === "https:" ? "wss" : "ws";
      socket = new WebSocket(`${protocol}://${window.location.host}/ws`);

      socket.onopen = () => {
        retry = 0;
        setConnected(true);
      };

      socket.onmessage = (event) => {
        let message: FeedMessage;
        try {
          message = JSON.parse(event.data) as FeedMessage;
        } catch {
          return;
        }
        switch (message.type) {
          case "hello":
            setState(message.state);
            setModel(message.model);
            setStats(message.state.stats);
            if (message.live) setLive(message.live);
            if (message.probe) setProbe(message.probe);
            setConnected(true);
            break;
          case "live":
            setLive(message);
            break;
          case "probe":
            setProbe(message);
            break;
          case "head":
            setHead(message);
            break;
          case "state":
            setState(message);
            break;
          case "status":
            setStats(message);
            setState((prev) =>
              prev ? { ...prev, stats: { ...prev.stats, ...message }, paused: prev.paused } : prev,
            );
            break;
          case "toast":
            notify(message.message, "success");
            break;
          case "error":
            notify(message.message, "error");
            break;
        }
      };

      socket.onclose = () => {
        setConnected(false);
        if (closed) return;
        retry = Math.min(retry + 1, 6);
        reconnectTimer = window.setTimeout(connect, 400 + retry * 500);
      };

      socket.onerror = () => {
        socket?.close();
      };
    };

    connect();
    return () => {
      closed = true;
      window.clearTimeout(reconnectTimer);
      socket?.close();
    };
  }, [notify]);

  return { connected, live, probe, head, state, model, stats, toasts, notify, dismiss };
}
