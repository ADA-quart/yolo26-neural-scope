import { useEffect } from "react";
import Launcher from "./screens/Launcher";
import LiveScreen from "./screens/LiveScreen";
import NeuralScreen from "./screens/NeuralScreen";

export default function App() {
  const path = window.location.pathname.replace(/\/+$/, "");

  useEffect(() => {
    document.title =
      path === "/live"
        ? "YOLO26 · 观察者"
        : path === "/neural"
          ? "YOLO26 · 神经显微镜"
          : "YOLO26 · 神经显微镜";
  }, [path]);

  if (path === "/live") return <LiveScreen />;
  if (path === "/neural") return <NeuralScreen />;
  return <Launcher />;
}
