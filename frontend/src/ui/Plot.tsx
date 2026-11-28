import { useEffect, useRef, useState, type ReactNode } from "react";

export const C = {
  bg: "#0c1014",
  grid: "#141b21",
  gridMajor: "#1d272e",
  axis: "#31404a",
  tick: "#a9b5bb",
  text: "#f6f8f9",
  twin: "#4fe3c1",
  real: "#f2b54a",
  simgan: "#6aa8ff",
  dcgan: "#ff7a59",
  alert: "#ff4d5e",
  dim: "#d3dbdf",
  wave: { P: "#7aa7ff", Q: "#b4e05a", R: "#ff5d73", S: "#f5a524", T: "#c38bff" } as Record<string, string>,
};

export interface Series {
  y: ArrayLike<number>;
  x?: ArrayLike<number>;
  color: string;
  width?: number;
  alpha?: number;
  dash?: number[];
  label?: string;
  glow?: boolean;
}

export interface Band {
  lo: ArrayLike<number>;
  hi: ArrayLike<number>;
  x?: ArrayLike<number>;
  color: string;
  alpha?: number;
}

export interface Marker {
  x: number;
  label: string;
  color: string;
}

interface Props {
  series: Series[];
  bands?: Band[];
  x: ArrayLike<number>; // default x for series without their own
  xDomain?: [number, number];
  yDomain?: [number, number];
  height?: number;
  markers?: Marker[];
  xLabel?: string;
  yTicks?: number;
  xStep?: number; // major grid step in x units
  paper?: boolean; // ECG-paper minor grid
  tooltip?: boolean;
  fmtY?: (v: number) => string;
  fmtX?: (v: number) => string;
  overlay?: ReactNode;
  padLeft?: number;
  bare?: boolean; // no axes or ticks (thumbnails)
}

function useSize(ref: React.RefObject<HTMLElement>) {
  const [w, setW] = useState(0);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(Math.floor(e.contentRect.width)));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, [ref]);
  return w;
}

export function niceTicks(lo: number, hi: number, n: number) {
  const span = hi - lo || 1;
  const raw = span / n;
  const mag = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) ?? raw;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-12; v += step) out.push(+v.toFixed(12));
  return out;
}

export function autoDomain(series: { y: ArrayLike<number> }[], pad = 0.08): [number, number] {
  let lo = Infinity;
  let hi = -Infinity;
  for (const s of series)
    for (let i = 0; i < s.y.length; i++) {
      const v = s.y[i];
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  if (!isFinite(lo)) return [-1, 1];
  const p = (hi - lo || 1) * pad;
  return [lo - p, hi + p];
}

const defaultFmtY = (v: number) => (Math.abs(v) < 1 ? v.toFixed(3) : v.toFixed(1));

export function TracePlot({
  series,
  bands = [],
  x,
  xDomain,
  yDomain,
  height = 260,
  markers = [],
  xLabel = "ms from R-peak",
  yTicks = 5,
  xStep = 200,
  paper = true,
  tooltip = true,
  fmtY = defaultFmtY,
  fmtX = (v) => `${Math.round(v)}`,
  overlay,
  padLeft = 46,
  bare = false,
}: Props) {
  const wrap = useRef<HTMLDivElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const width = useSize(wrap);
  const [hover, setHover] = useState<number | null>(null);

  const xd: [number, number] = xDomain ?? [x[0], x[x.length - 1]];
  const yd = yDomain ?? autoDomain([...series, ...bands.flatMap((b) => [{ y: b.lo }, { y: b.hi }])]);
  const pad = bare ? { l: 2, r: 2, t: 4, b: 4 } : { l: padLeft, r: 12, t: 12, b: 26 };

  useEffect(() => {
    const cv = canvas.current;
    if (!cv || width === 0) return;
    const dpr = window.devicePixelRatio || 1;
    cv.width = width * dpr;
    cv.height = height * dpr;
    const g = cv.getContext("2d")!;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, width, height);
    const W = width - pad.l - pad.r;
    const H = height - pad.t - pad.b;
    const sx = (v: number) => pad.l + ((v - xd[0]) / (xd[1] - xd[0])) * W;
    const sy = (v: number) => pad.t + (1 - (v - yd[0]) / (yd[1] - yd[0])) * H;

    // grid
    g.lineWidth = 1;
    if (paper) {
      g.strokeStyle = C.grid;
      g.beginPath();
      const minor = xStep / 5;
      for (let v = Math.ceil(xd[0] / minor) * minor; v <= xd[1]; v += minor) {
        const px = Math.round(sx(v)) + 0.5;
        g.moveTo(px, pad.t);
        g.lineTo(px, pad.t + H);
      }
      g.stroke();
    }
    g.strokeStyle = C.gridMajor;
    g.beginPath();
    for (let v = Math.ceil(xd[0] / xStep) * xStep; v <= xd[1]; v += xStep) {
      const px = Math.round(sx(v)) + 0.5;
      g.moveTo(px, pad.t);
      g.lineTo(px, pad.t + H);
    }
    const yt = niceTicks(yd[0], yd[1], yTicks);
    for (const v of yt) {
      const py = Math.round(sy(v)) + 0.5;
      g.moveTo(pad.l, py);
      g.lineTo(pad.l + W, py);
    }
    g.stroke();

    // zero line
    if (yd[0] < 0 && yd[1] > 0) {
      g.strokeStyle = C.axis;
      g.setLineDash([2, 3]);
      g.beginPath();
      const py = Math.round(sy(0)) + 0.5;
      g.moveTo(pad.l, py);
      g.lineTo(pad.l + W, py);
      g.stroke();
      g.setLineDash([]);
    }

    // ticks
    if (!bare) {
    g.fillStyle = C.tick;
    g.font = "11px 'JetBrains Mono', monospace";
    g.textAlign = "right";
    g.textBaseline = "middle";
    for (const v of yt) g.fillText(fmtY(v), pad.l - 7, sy(v));
    g.textAlign = "center";
    g.textBaseline = "top";
    const labelW = xLabel ? g.measureText(xLabel.toUpperCase()).width + 16 : 0;
    for (let v = Math.ceil(xd[0] / xStep) * xStep; v <= xd[1]; v += xStep) {
      const px = sx(v);
      if (px + 14 > pad.l + W - labelW) continue; // keep clear of the axis label
      g.fillText(fmtX(v), px, pad.t + H + 7);
    }
    g.textAlign = "right";
    g.fillStyle = "#7d8a91";
    g.fillText(xLabel.toUpperCase(), pad.l + W, pad.t + H + 7);
    }

    // markers
    for (const m of markers) {
      if (m.x < xd[0] || m.x > xd[1]) continue;
      const px = Math.round(sx(m.x)) + 0.5;
      g.strokeStyle = m.color;
      g.globalAlpha = 0.45;
      g.setLineDash([1, 3]);
      g.beginPath();
      g.moveTo(px, pad.t + 12);
      g.lineTo(px, pad.t + H);
      g.stroke();
      g.setLineDash([]);
      g.globalAlpha = 1;
      g.fillStyle = m.color;
      g.textAlign = "center";
      g.textBaseline = "top";
      g.font = "600 11px 'JetBrains Mono', monospace";
      g.fillText(m.label, px, pad.t);
    }

    g.save();
    g.beginPath();
    g.rect(pad.l, pad.t, W, H);
    g.clip();

    for (const b of bands) {
      const bx = b.x ?? x;
      g.globalAlpha = b.alpha ?? 0.16;
      g.fillStyle = b.color;
      g.beginPath();
      for (let i = 0; i < b.hi.length; i++) g.lineTo(sx(bx[i]), sy(b.hi[i]));
      for (let i = b.lo.length - 1; i >= 0; i--) g.lineTo(sx(bx[i]), sy(b.lo[i]));
      g.closePath();
      g.fill();
    }
    g.globalAlpha = 1;

    g.lineJoin = "round";
    g.lineCap = "round";
    for (const s of series) {
      const sxs = s.x ?? x;
      g.strokeStyle = s.color;
      g.globalAlpha = s.alpha ?? 1;
      g.lineWidth = s.width ?? 1.5;
      g.setLineDash(s.dash ?? []);
      if (s.glow) {
        g.shadowColor = s.color;
        g.shadowBlur = 8;
      }
      g.beginPath();
      for (let i = 0; i < s.y.length; i++) {
        const px = sx(sxs[i]);
        const py = sy(s.y[i]);
        if (i === 0) g.moveTo(px, py);
        else g.lineTo(px, py);
      }
      g.stroke();
      g.shadowBlur = 0;
    }
    g.setLineDash([]);
    g.globalAlpha = 1;

    if (hover !== null) {
      const px = Math.round(sx(hover)) + 0.5;
      g.strokeStyle = "#d3dbdf";
      g.globalAlpha = 0.5;
      g.beginPath();
      g.moveTo(px, pad.t);
      g.lineTo(px, pad.t + H);
      g.stroke();
      g.globalAlpha = 1;
      for (const s of series) {
        if (!s.label) continue;
        const i = nearest(s.x ?? x, hover);
        g.fillStyle = s.color;
        g.fillRect(sx((s.x ?? x)[i]) - 2.5, sy(s.y[i]) - 2.5, 5, 5);
      }
    }
    g.restore();
  });

  const onMove = (e: React.PointerEvent) => {
    if (!tooltip) return;
    const r = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const px = e.clientX - r.left;
    const W = width - pad.l - pad.r;
    if (px < pad.l || px > pad.l + W) return setHover(null);
    setHover(xd[0] + ((px - pad.l) / W) * (xd[1] - xd[0]));
  };

  const labelled = series.filter((s) => s.label);
  let tip: ReactNode = null;
  if (hover !== null && labelled.length) {
    const W = width - pad.l - pad.r;
    const left = pad.l + ((hover - xd[0]) / (xd[1] - xd[0])) * W;
    const flip = left > width * 0.62;
    tip = (
      <div className="plot-tip" style={{ top: pad.t + 4, left: flip ? undefined : left + 10, right: flip ? width - left + 10 : undefined }}>
        <div className="t">t = {fmtX(hover)} ms</div>
        {labelled.map((s, k) => {
          const i = nearest(s.x ?? x, hover);
          return (
            <div className="r" key={k}>
              <span className="swatch" style={{ background: s.color }} />
              <span style={{ color: "#d3dbdf", minWidth: 70 }}>{s.label}</span>
              <span>{s.y[i].toFixed(5)}</span>
            </div>
          );
        })}
      </div>
    );
  }

  return (
    <div className="plot" ref={wrap} style={{ height }} onPointerMove={onMove} onPointerLeave={() => setHover(null)}>
      <canvas ref={canvas} />
      {tip}
      {overlay}
    </div>
  );
}

function nearest(xs: ArrayLike<number>, v: number) {
  let best = 0;
  let d = Infinity;
  for (let i = 0; i < xs.length; i++) {
    const e = Math.abs(xs[i] - v);
    if (e < d) {
      d = e;
      best = i;
    }
  }
  return best;
}

export function Legend({ items }: { items: { color: string; label: string; dash?: boolean }[] }) {
  return (
    <div className="plot-legend">
      {items.map((it) => (
        <span key={it.label} className="legend-row">
          <span
            className="swatch"
            style={
              it.dash
                ? { background: `repeating-linear-gradient(90deg, ${it.color} 0 4px, transparent 4px 7px)` }
                : { background: it.color }
            }
          />
          {it.label}
        </span>
      ))}
    </div>
  );
}
