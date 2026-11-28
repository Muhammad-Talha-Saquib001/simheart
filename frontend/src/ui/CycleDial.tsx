import { useRef } from "react";
import { N_SAMPLES, PHASE, WAVES, type Params, type Ranges, type Wave } from "../engine/ecgsyn";
import { C } from "./Plot";

/**
 * The beat cycle seen from above: one lap of the circle is one beat (R at the top).
 * Each wave sits at its angle θ with an arc of ±2b; drag a handle to move the wave.
 * The beat itself is drawn round the ring (outward = positive).
 */
export function CycleDial({
  params,
  limits,
  beat,
  onTheta,
  size = 300,
}: {
  params: Params;
  limits: Ranges;
  beat: ArrayLike<number>;
  onTheta: (w: Wave, deg: number) => void;
  size?: number;
}) {
  const svg = useRef<SVGSVGElement>(null);
  const drag = useRef<Wave | null>(null);
  const c = size / 2;
  const r = size * 0.33;
  // θ = 0 at the top, clockwise
  const pt = (deg: number, rad: number) => {
    const a = ((deg - 90) * Math.PI) / 180;
    return [c + rad * Math.cos(a), c + rad * Math.sin(a)] as const;
  };
  const arc = (d0: number, d1: number, rad: number) => {
    const [x0, y0] = pt(d0, rad);
    const [x1, y1] = pt(d1, rad);
    const large = d1 - d0 > 180 ? 1 : 0;
    return `M${x0},${y0} A${rad},${rad} 0 ${large} 1 ${x1},${y1}`;
  };

  let maxAbs = 1e-9;
  for (let i = 0; i < beat.length; i++) maxAbs = Math.max(maxAbs, Math.abs(beat[i]));
  const k = (size * 0.13) / maxAbs;
  let path = "";
  for (let i = 0; i < N_SAMPLES; i++) {
    const deg = (Math.atan2(PHASE.ys[i], PHASE.xs[i]) * 180) / Math.PI;
    const [x, y] = pt(deg, r * 0.62 + beat[i] * k);
    path += `${i ? "L" : "M"}${x.toFixed(2)},${y.toFixed(2)}`;
  }

  const onMove = (e: React.PointerEvent) => {
    const w = drag.current;
    if (!w || !svg.current) return;
    const box = svg.current.getBoundingClientRect();
    const x = ((e.clientX - box.left) / box.width) * size - c;
    const y = ((e.clientY - box.top) / box.height) * size - c;
    let deg = (Math.atan2(y, x) * 180) / Math.PI + 90;
    if (deg > 180) deg -= 360;
    const [lo, hi] = limits[w].theta_deg;
    // T can sit past 180°: unwrap towards its range
    if (deg < lo && deg + 360 <= hi) deg += 360;
    onTheta(w, Math.max(lo, Math.min(hi, +deg.toFixed(2))));
  };

  const ticks = Array.from({ length: 72 }, (_, i) => i * 5);

  return (
    <svg
      ref={svg}
      viewBox={`0 0 ${size} ${size}`}
      style={{ width: "100%", maxWidth: size, display: "block", margin: "0 auto", touchAction: "none" }}
      onPointerMove={onMove}
      onPointerUp={() => (drag.current = null)}
      onPointerLeave={() => (drag.current = null)}
    >
      <circle cx={c} cy={c} r={r} fill="none" stroke="#222d35" />
      <circle cx={c} cy={c} r={r * 0.62} fill="none" stroke="#141b21" strokeDasharray="2 3" />
      {ticks.map((d) => {
        const major = d % 30 === 0;
        const [x0, y0] = pt(d, r + 4);
        const [x1, y1] = pt(d, r + (major ? 11 : 7));
        return <line key={d} x1={x0} y1={y0} x2={x1} y2={y1} stroke={major ? "#31404a" : "#1d272e"} />;
      })}
      {[0, 90, 180, 270].map((d) => {
        const near = WAVES.some((w) => Math.abs((((params[w].theta_deg - d) % 360) + 540) % 360 - 180) < 22);
        if (near) return null;
        const [x, y] = pt(d, r + 22);
        const label = d > 180 ? d - 360 : d;
        return (
          <text key={d} x={x} y={y} fill="#7d8a91" fontSize="10.5" fontFamily="JetBrains Mono" textAnchor="middle" dominantBaseline="middle">
            {label}°
          </text>
        );
      })}
      <path d={path} fill="none" stroke={C.twin} strokeWidth="1.4" style={{ filter: `drop-shadow(0 0 4px ${C.twin}66)` }} />
      {WAVES.map((w) => {
        const { theta_deg, b } = params[w];
        const half = Math.min(170, ((2 * b * 180) / Math.PI));
        const [hx, hy] = pt(theta_deg, r);
        const [lx, ly] = pt(theta_deg, r + 36);
        return (
          <g key={w}>
            <path d={arc(theta_deg - half, theta_deg + half, r)} fill="none" stroke={C.wave[w]} strokeWidth="5" strokeOpacity="0.28" />
            <line x1={c} y1={c} x2={hx} y2={hy} stroke={C.wave[w]} strokeOpacity="0.18" />
            <g
              style={{ cursor: "grab" }}
              onPointerDown={(e) => {
                (e.target as Element).setPointerCapture?.(e.pointerId);
                drag.current = w;
              }}
            >
              <circle cx={hx} cy={hy} r="11" fill="transparent" />
              <rect x={hx - 5} y={hy - 5} width="10" height="10" fill={C.wave[w]} stroke="#07090b" strokeWidth="2" transform={`rotate(45 ${hx} ${hy})`} />
            </g>
            <text x={lx} y={ly} fill={C.wave[w]} fontSize="12" fontWeight="600" fontFamily="JetBrains Mono" textAnchor="middle" dominantBaseline="middle">
              {w}
            </text>
          </g>
        );
      })}
      <text x={c} y={c - 6} fill="#a9b5bb" fontSize="10.5" fontFamily="JetBrains Mono" textAnchor="middle" letterSpacing="1.5">
        ONE LAP
      </text>
      <text x={c} y={c + 8} fill="#a9b5bb" fontSize="10.5" fontFamily="JetBrains Mono" textAnchor="middle" letterSpacing="1.5">
        = ONE BEAT
      </text>
    </svg>
  );
}
