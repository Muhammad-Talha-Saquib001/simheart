import { useMemo, useState } from "react";
import { useTwin } from "../backend";
import { FIELDS, MS_PER_RAD, TIME_MS, WAVES, cloneParams, decompose, modelIntervals, simulate, stats, type Field, type Params, type Wave } from "../engine/ecgsyn";
import type { BeatClass } from "../engine/rhythm";
import { CycleDial } from "../ui/CycleDial";
import { C, Legend, TracePlot } from "../ui/Plot";
import { Panel, Readout, Seg, Slider } from "../ui/controls";

const WAVE_NAME: Record<Wave, string> = {
  P: "atrial depolarisation",
  Q: "septal",
  R: "ventricular depolarisation",
  S: "late ventricular",
  T: "repolarisation",
};
const FIELD_LABEL: Record<Field, string> = { a: "a", b: "b", theta_deg: "θ" };
const DIGITS: Record<Field, number> = { a: 3, b: 3, theta_deg: 1 };

export function WaveControls({
  params,
  reference,
  onChange,
  cls,
  compact,
}: {
  params: Params;
  reference: Params;
  onChange: (w: Wave, f: Field, v: number) => void;
  cls: BeatClass;
  compact?: boolean;
}) {
  const twin = useTwin();
  const band = twin.bands[cls];
  return (
    <div style={{ display: "grid", gap: 6 }}>
      {WAVES.map((w) => (
        <div className="wave-card" key={w}>
          <div className="wave-card-head">
            <span className="wave-letter" style={{ background: C.wave[w] }}>
              {w}
            </span>
            {!compact && <span className="wave-name">{WAVE_NAME[w]}</span>}
          </div>
          {FIELDS.map((f) => {
            const [lo, hi] = twin.limits[w][f];
            return (
              <Slider
                key={f}
                k={FIELD_LABEL[f]}
                value={params[w][f]}
                min={lo}
                max={hi}
                band={band?.[w][f]}
                reference={reference[w][f]}
                color={C.wave[w]}
                digits={DIGITS[f]}
                onChange={(v) => onChange(w, f, v)}
              />
            );
          })}
        </div>
      ))}
    </div>
  );
}

export function Morphology() {
  const twin = useTwin();
  const [cls, setCls] = useState<BeatClass>("V");
  const [params, setParams] = useState<Params>(() => cloneParams(twin.presets.V));
  const [showParts, setShowParts] = useState(true);
  const preset = twin.presets[cls];

  const z = useMemo(() => simulate(params), [params]);
  const ref = useMemo(() => simulate(preset), [preset]);
  const parts = useMemo(() => decompose(params), [params]);
  const st = stats(z);
  const iv = modelIntervals(params);
  const ivRef = modelIntervals(preset);

  const pick = (c: BeatClass) => {
    setCls(c);
    setParams(cloneParams(twin.presets[c]));
  };
  const change = (w: Wave, f: Field, v: number) =>
    setParams((p) => {
      const n = cloneParams(p);
      n[w][f] = v;
      return n;
    });

  const jitter = () => {
    const band = twin.bands[cls];
    const n = cloneParams(preset);
    for (const w of WAVES)
      for (const f of FIELDS) {
        if (band) {
          const [lo, hi] = band[w][f];
          n[w][f] = lo + Math.random() * (hi - lo);
        } else n[w][f] *= 1 + (Math.random() - 0.5) * 0.3;
      }
    setParams(n);
  };

  const markers = WAVES.map((w) => ({ x: params[w].theta_deg * (Math.PI / 180) * MS_PER_RAD, label: w, color: C.wave[w] }));
  const delta = (a: number, b: number) => {
    const d = a - b;
    if (Math.abs(d) < 0.5) return "preset";
    return `${d > 0 ? "+" : ""}${d.toFixed(0)} ms vs preset`;
  };

  return (
    <div className="module">
      <div className="module-head">
        <h1>
          <span>02</span>MORPHOLOGY LAB
        </h1>
        <p>
          Every beat is set by 15 numbers: an amplitude <b>a</b>, a width <b>b</b> and an angle <b>θ</b> for each of the P, Q, R, S
          and T waves. Change them and the simulated beat updates live.
        </p>
        <div className="tools">
          <Seg value={cls} options={["N", "S", "V", "F"] as const} onChange={pick} />
          <button className="btn ghost" onClick={jitter} title="Draw every parameter at random within the class's plausible range">
            Sample class
          </button>
          <button className="btn ghost" onClick={() => setParams(cloneParams(preset))}>
            Reset
          </button>
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "310px minmax(0, 1fr) 300px" }} data-stack>
        <Panel
          code="A"
          title="Wave parameters"
          right={<span className="hint">{twin.bands[cls] ? "band: 1–99 pct of fits" : "dbl-click resets"}</span>}
        >
          <WaveControls params={params} reference={preset} onChange={change} cls={cls} />
        </Panel>

        <div className="grid" style={{ alignContent: "start" }}>
          <Panel
            code="B"
            title={`Simulated beat · class ${cls}`}
            right={<Legend items={[{ color: C.twin, label: "twin" }, { color: C.tick, label: "class preset", dash: true }]} />}
            flush
          >
            <div style={{ padding: "6px 6px 0" }}>
              <TracePlot
                x={TIME_MS}
                height={300}
                markers={markers}
                series={[
                  { y: ref, color: C.tick, width: 1.2, dash: [4, 4], label: "preset" },
                  { y: z, color: C.twin, width: 2, glow: true, label: "twin" },
                ]}
              />
            </div>
            <div className="readouts">
              <Readout k="R peak" v={(st.max * 1000).toFixed(1)} unit="×10⁻³" sub={`at ${Math.round(TIME_MS[st.iMax])} ms`} />
              <Readout k="Span" v={(st.span * 1000).toFixed(1)} unit="×10⁻³" sub="max − min, sim. units" />
              <Readout k="PR (model)" v={Math.round(iv.pr)} unit="ms" sub={delta(iv.pr, ivRef.pr)} />
              <Readout k="QRS (model)" v={Math.round(iv.qrs)} unit="ms" sub={delta(iv.qrs, ivRef.qrs)} color={iv.qrs > 120 ? C.real : undefined} />
              <Readout k="QT (model)" v={Math.round(iv.qt)} unit="ms" sub={delta(iv.qt, ivRef.qt)} />
            </div>
          </Panel>

          <Panel
            code="C"
            title="Wave decomposition"
            right={
              <>
                <Legend items={WAVES.map((w) => ({ color: C.wave[w], label: w }))} />
                <button className="chip" onClick={() => setShowParts((s) => !s)}>
                  {showParts ? "hide" : "show"}
                </button>
              </>
            }
            flush
          >
            {showParts && (
              <div style={{ padding: "6px 6px 0" }}>
                <TracePlot
                  x={TIME_MS}
                  height={170}
                  series={WAVES.map((w, i) => ({ y: parts[i], color: C.wave[w], width: 1.5, label: w }))}
                />
              </div>
            )}
            <div className="panel-body" style={{ borderTop: "1px solid var(--line)" }}>
              <div className="note">
                The model is linear in z, so the beat is exactly the sum of each wave's own response (plus a tiny baseline wander).
                Times use the model's clock: one lap of 2π = 600 ms, so 1° ≈ {(MS_PER_RAD * (Math.PI / 180)).toFixed(2)} ms.
              </div>
            </div>
          </Panel>
        </div>

        <div className="grid" style={{ alignContent: "start" }}>
          <Panel code="D" title="Cycle dial" right={<span className="hint">drag ◆ to move θ</span>}>
            <CycleDial params={params} limits={twin.limits} beat={z} onTheta={(w, d) => change(w, "theta_deg", d)} />
          </Panel>
          <Panel code="E" title="Parameter vector" flush>
            <table className="data">
              <thead>
                <tr>
                  <th />
                  <th className="num">a</th>
                  <th className="num">b</th>
                  <th className="num">θ°</th>
                </tr>
              </thead>
              <tbody>
                {WAVES.map((w) => (
                  <tr key={w}>
                    <td style={{ color: C.wave[w], fontWeight: 600 }}>{w}</td>
                    {FIELDS.map((f) => (
                      <td key={f} className="num" style={{ color: Math.abs(params[w][f] - preset[w][f]) > 1e-6 ? C.twin : undefined }}>
                        {params[w][f].toFixed(DIGITS[f])}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
            <div className="panel-body" style={{ borderTop: "1px solid var(--line)" }}>
              <button className="btn ghost" style={{ width: "100%", justifyContent: "center" }} onClick={() => navigator.clipboard?.writeText(JSON.stringify(params, null, 2))}>
                Copy as JSON (api.simulate)
              </button>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}
