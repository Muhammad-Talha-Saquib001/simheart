import { useEffect, useRef } from "react";
import { N_SAMPLES, PHASE, WAVES, type Params } from "../engine/ecgsyn";
import { C } from "./Plot";

export interface PortraitState {
  values: ArrayLike<number>;
  index: number | null; // sample under the cursor (animated point), null for none
  params: Params;
  color: string;
  ghost?: ArrayLike<number> | null;
}

/**
 * The model's state space in 3D: (x, y) runs round the unit circle once per beat, z is
 * the ECG value lifted off that circle. Drag to rotate.
 */
export function PhasePortrait({ source, height = 300, zScale = 22 }: { source: React.MutableRefObject<PortraitState>; height?: number; zScale?: number }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const view = useRef({ yaw: -0.65, pitch: 0.52, drag: null as null | { x: number; y: number; yaw: number; pitch: number }, idle: 0 });

  useEffect(() => {
    let raf = 0;
    const cv = canvas.current!;
    const draw = () => {
      raf = requestAnimationFrame(draw);
      const width = cv.clientWidth;
      if (!width) return;
      const dpr = window.devicePixelRatio || 1;
      if (cv.width !== width * dpr || cv.height !== height * dpr) {
        cv.width = width * dpr;
        cv.height = height * dpr;
      }
      const g = cv.getContext("2d")!;
      g.setTransform(dpr, 0, 0, dpr, 0, 0);
      g.clearRect(0, 0, width, height);
      const v = view.current;
      if (!v.drag) {
        v.idle++;
        if (v.idle > 90) v.yaw += 0.0025;
      }
      const { values, index, params, color, ghost } = source.current;
      const cy = Math.cos(v.yaw);
      const sy = Math.sin(v.yaw);
      const cp = Math.cos(v.pitch);
      const sp = Math.sin(v.pitch);
      const scale = Math.min(width, height * 1.5) * 0.34;
      const ox = width / 2;
      const oy = height * 0.6;
      const proj = (x: number, y: number, z: number) => {
        // yaw round the vertical (z) axis, then pitch
        const X = x * cy - y * sy;
        const Y = x * sy + y * cy;
        const depth = Y * cp + z * sp;
        const up = z * cp - Y * sp;
        const persp = 1 / (1 + depth * 0.12);
        return [ox + X * scale * persp, oy - up * scale * persp, depth] as const;
      };
      const zOf = (val: number) => val * zScale;

      // floor rings
      g.lineWidth = 1;
      for (const [r, a] of [[1, 0.9], [1.35, 0.35], [0.65, 0.35]] as const) {
        g.strokeStyle = r === 1 ? "#2a3740" : "#172027";
        g.globalAlpha = a;
        g.beginPath();
        for (let i = 0; i <= 96; i++) {
          const t = (i / 96) * Math.PI * 2;
          const [px, py] = proj(r * Math.cos(t), r * Math.sin(t), 0);
          if (i === 0) g.moveTo(px, py);
          else g.lineTo(px, py);
        }
        g.stroke();
      }
      g.globalAlpha = 1;

      // axis cross
      g.strokeStyle = "#1d272e";
      g.beginPath();
      for (const [x, y] of [[-1.45, 0], [1.45, 0], [0, -1.45], [0, 1.45]] as const) {
        const [a1, b1] = proj(0, 0, 0);
        const [a2, b2] = proj(x, y, 0);
        g.moveTo(a1, b1);
        g.lineTo(a2, b2);
      }
      g.stroke();

      // wave spokes and labels
      g.font = "600 10px 'IBM Plex Mono', monospace";
      g.textAlign = "center";
      g.textBaseline = "middle";
      for (const w of WAVES) {
        const t = (params[w].theta_deg * Math.PI) / 180;
        const [x1, y1] = proj(Math.cos(t) * 1.0, Math.sin(t) * 1.0, 0);
        const [x2, y2] = proj(Math.cos(t) * 1.3, Math.sin(t) * 1.3, 0);
        g.strokeStyle = C.wave[w];
        g.globalAlpha = 0.6;
        g.beginPath();
        g.moveTo(x1, y1);
        g.lineTo(x2, y2);
        g.stroke();
        g.globalAlpha = 1;
        const [lx, ly] = proj(Math.cos(t) * 1.45, Math.sin(t) * 1.45, 0);
        g.fillStyle = C.wave[w];
        g.fillText(w, lx, ly);
      }

      // curtain: drop lines from the trajectory to the floor
      g.strokeStyle = color;
      g.lineWidth = 1;
      for (let i = 0; i < N_SAMPLES; i += 3) {
        const [ax, ay] = proj(PHASE.xs[i], PHASE.ys[i], 0);
        const [bx, by] = proj(PHASE.xs[i], PHASE.ys[i], zOf(values[i]));
        g.globalAlpha = 0.09;
        g.beginPath();
        g.moveTo(ax, ay);
        g.lineTo(bx, by);
        g.stroke();
      }
      g.globalAlpha = 1;

      if (ghost) {
        g.strokeStyle = "#5d6b73";
        g.setLineDash([3, 3]);
        g.beginPath();
        for (let i = 0; i < N_SAMPLES; i++) {
          const [px, py] = proj(PHASE.xs[i], PHASE.ys[i], zOf(ghost[i]));
          if (i === 0) g.moveTo(px, py);
          else g.lineTo(px, py);
        }
        g.stroke();
        g.setLineDash([]);
      }

      // trajectory
      g.lineWidth = 1.8;
      g.strokeStyle = color;
      g.shadowColor = color;
      g.shadowBlur = 10;
      g.beginPath();
      for (let i = 0; i < N_SAMPLES; i++) {
        const [px, py] = proj(PHASE.xs[i], PHASE.ys[i], zOf(values[i]));
        if (i === 0) g.moveTo(px, py);
        else g.lineTo(px, py);
      }
      g.stroke();
      g.shadowBlur = 0;

      // current state
      if (index !== null) {
        const i = Math.max(0, Math.min(N_SAMPLES - 1, index));
        const [fx, fy] = proj(PHASE.xs[i], PHASE.ys[i], 0);
        const [px, py] = proj(PHASE.xs[i], PHASE.ys[i], zOf(values[i]));
        g.strokeStyle = "#d9e1e5";
        g.globalAlpha = 0.5;
        g.beginPath();
        g.moveTo(fx, fy);
        g.lineTo(px, py);
        g.stroke();
        g.globalAlpha = 1;
        g.fillStyle = "#d9e1e5";
        g.fillRect(fx - 1.5, fy - 1.5, 3, 3);
        g.shadowColor = color;
        g.shadowBlur = 14;
        g.fillStyle = "#ffffff";
        g.beginPath();
        g.arc(px, py, 3.2, 0, Math.PI * 2);
        g.fill();
        g.shadowBlur = 0;
        // readout
        g.font = "10px 'IBM Plex Mono', monospace";
        g.textAlign = "left";
        g.fillStyle = "#9aa7ae";
        const th = (Math.atan2(PHASE.ys[i], PHASE.xs[i]) * 180) / Math.PI;
        g.fillText(`θ ${th.toFixed(1).padStart(6)}°`, 12, 16);
        g.fillText(`z ${values[i].toFixed(4).padStart(8)}`, 12, 30);
      }
      g.font = "9px 'IBM Plex Mono', monospace";
      g.textAlign = "right";
      g.fillStyle = "#3a464d";
      g.fillText("DRAG TO ROTATE", width - 10, height - 10);
    };
    draw();
    return () => cancelAnimationFrame(raf);
  }, [height, source, zScale]);

  return (
    <canvas
      ref={canvas}
      style={{ width: "100%", height, display: "block", cursor: "grab", touchAction: "none" }}
      onPointerDown={(e) => {
        (e.target as HTMLElement).setPointerCapture(e.pointerId);
        const v = view.current;
        v.drag = { x: e.clientX, y: e.clientY, yaw: v.yaw, pitch: v.pitch };
      }}
      onPointerMove={(e) => {
        const v = view.current;
        if (!v.drag) return;
        v.yaw = v.drag.yaw + (e.clientX - v.drag.x) * 0.008;
        v.pitch = Math.max(-0.2, Math.min(1.35, v.drag.pitch + (e.clientY - v.drag.y) * 0.006));
      }}
      onPointerUp={() => {
        view.current.drag = null;
        view.current.idle = 0;
      }}
    />
  );
}
