import { useRef, type CSSProperties, type ReactNode } from "react";

export function Panel({
  title,
  code,
  right,
  children,
  flush,
  style,
  className = "",
}: {
  title: ReactNode;
  code?: string;
  right?: ReactNode;
  children: ReactNode;
  flush?: boolean;
  style?: CSSProperties;
  className?: string;
}) {
  return (
    <section className={`panel ${className}`} style={style}>
      <header className="panel-head">
        <h2 className="panel-title" data-code={code}>
          {title}
        </h2>
        {right && <div className="right">{right}</div>}
      </header>
      <div className={`panel-body ${flush ? "flush" : ""}`} style={{ flex: 1 }}>
        {children}
      </div>
    </section>
  );
}

export function Seg<T extends string | number>({
  value,
  options,
  onChange,
  labels,
}: {
  value: T;
  options: readonly T[];
  onChange: (v: T) => void;
  labels?: Partial<Record<T, string>>;
}) {
  return (
    <div className="seg" role="radiogroup">
      {options.map((o) => (
        <button key={String(o)} role="radio" aria-checked={o === value} className={o === value ? "on" : ""} onClick={() => onChange(o)}>
          {labels?.[o] ?? String(o)}
        </button>
      ))}
    </div>
  );
}

export function Readout({ k, v, unit, sub, color, help }: { k: string; v: ReactNode; unit?: string; sub?: ReactNode; color?: string; help?: string }) {
  return (
    <div className="readout" title={help}>
      <div className="readout-k">{k}</div>
      <div className="readout-v" style={{ color }}>
        {v}
        {unit && <small>{unit}</small>}
      </div>
      {sub && <div className="readout-s">{sub}</div>}
    </div>
  );
}

/**
 * Precision slider: full drag range [min, max], an optional highlighted band (the class's
 * plausible values), and a reference tick (the preset).
 */
export function Slider({
  k,
  label,
  value,
  min,
  max,
  onChange,
  band,
  reference,
  color,
  digits = 3,
  step,
}: {
  k: string;
  label?: string; // plain-language name shown next to the symbol
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
  band?: [number, number];
  reference?: number;
  color?: string;
  digits?: number;
  step?: number;
}) {
  const track = useRef<HTMLDivElement>(null);
  const span = max - min || 1;
  const pct = (v: number) => `${Math.max(0, Math.min(100, ((v - min) / span) * 100))}%`;
  const s = step ?? span / 200;

  const fromEvent = (e: React.PointerEvent) => {
    const r = track.current!.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width));
    onChange(+(min + f * span).toFixed(6));
  };

  const zero = Math.max(min, Math.min(max, 0));
  const fillFrom = Math.min(value, zero);
  const fillTo = Math.max(value, zero);
  const changed = reference !== undefined && Math.abs(value - reference) > span * 1e-4;

  return (
    <div className="slider" style={{ "--c": color } as CSSProperties}>
      <span className="slider-k">
        <b>{k}</b>
        {label}
      </span>
      <div
        className="track"
        ref={track}
        tabIndex={0}
        role="slider"
        aria-label={label ? `${label} (${k})` : k}
        aria-valuemin={min}
        aria-valuemax={max}
        aria-valuenow={value}
        onPointerDown={(e) => {
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
          fromEvent(e);
        }}
        onPointerMove={(e) => e.buttons === 1 && fromEvent(e)}
        onDoubleClick={() => reference !== undefined && onChange(reference)}
        onKeyDown={(e) => {
          const m = e.shiftKey ? 10 : 1;
          if (e.key === "ArrowRight" || e.key === "ArrowUp") onChange(Math.min(max, value + s * m));
          else if (e.key === "ArrowLeft" || e.key === "ArrowDown") onChange(Math.max(min, value - s * m));
          else return;
          e.preventDefault();
        }}
      >
        <div className="track-line" />
        {band && <div className="track-band" style={{ left: pct(band[0]), width: `calc(${pct(band[1])} - ${pct(band[0])})` }} />}
        <div className="track-fill" style={{ left: pct(fillFrom), width: `calc(${pct(fillTo)} - ${pct(fillFrom)})` }} />
        {reference !== undefined && <div className="track-ref" style={{ left: pct(reference) }} />}
        <div className="track-thumb" style={{ left: pct(value) }} />
      </div>
      <span className={`slider-v ${changed ? "changed" : ""}`}>{value.toFixed(digits)}</span>
    </div>
  );
}

export function Led({ state, pulse }: { state: "on" | "warn" | "off" | "idle"; pulse?: boolean }) {
  return <span className={`led ${state} ${pulse ? "pulse" : ""}`} />;
}

export function Progress({ value }: { value: number }) {
  return (
    <div className="progress">
      <div style={{ width: `${Math.max(0, Math.min(1, value)) * 100}%` }} />
    </div>
  );
}

/** A short numbered how-to: the steps are a real order of use. */
export function Steps({ items }: { items: string[] }) {
  return (
    <ol className="steps" aria-label="How to use this screen">
      <li>
        <span className="k">How to use</span>
      </li>
      {items.map((t, i) => (
        <li key={i}>
          <span className="n">{i + 1}</span>
          {t}
        </li>
      ))}
    </ol>
  );
}
