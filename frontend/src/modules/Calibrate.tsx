import { useEffect, useMemo, useRef, useState } from "react";
import { backend, NOTICE, useTwin, type BeatsResponse } from "../backend";
import { TIME_MS, TYPICAL, WAVES, cloneParams, modelIntervals, simulate, type Field, type Params, type Wave } from "../engine/ecgsyn";
import { Calibrator, START_NAMES } from "../engine/fitting";
import { alignmentWarning, importBeat, MAX_BEAT_FILE_BYTES } from "../engine/importBeat";
import { mulberry32 } from "../engine/rhythm";
import { C, Legend, TracePlot } from "../ui/Plot";
import { Panel, Progress, Readout, Seg, Steps } from "../ui/controls";
import { WaveControls } from "./Morphology";

type Cls = "S" | "V" | "F";

interface Target {
  values: Float64Array;
  label: string;
  real: boolean;
}

const SCENARIOS: { id: string; label: string; apply: (p: Params) => void }[] = [
  { id: "t2", label: "T wave ×2", apply: (p) => (p.T.a *= 2) },
  { id: "tinv", label: "Inverted T", apply: (p) => (p.T.a *= -1) },
  { id: "nop", label: "No P wave", apply: (p) => (p.P.a = 0) },
  { id: "wqrs", label: "Wide QRS", apply: (p) => ["Q", "R", "S"].forEach((w) => (p[w as Wave].b *= 1.6)) },
  { id: "tallr", label: "Tall R", apply: (p) => (p.R.a *= 1.4) },
  { id: "deeps", label: "Deep S", apply: (p) => (p.S.a *= 2) },
  { id: "lqt", label: "Late T", apply: (p) => (p.T.theta_deg += 25) },
];

const START_COLOR = [C.dim, C.wave.S, C.alert, C.real, C.wave.P];

/** Offline stand-in targets: a perturbed simulated beat plus noise. Clearly not real data. */
function demoTargets(cls: Cls, n: number): Target[] {
  const rand = mulberry32(cls.charCodeAt(0) * 97 + 3);
  return Array.from({ length: n }, (_, k) => {
    const p = cloneParams(TYPICAL[cls]);
    for (const w of WAVES) {
      p[w].a *= 1 + (rand() - 0.5) * 0.5;
      p[w].b *= 1 + (rand() - 0.5) * 0.4;
      p[w].theta_deg += (rand() - 0.5) * 10;
    }
    const z = simulate(p, 0);
    for (let i = 0; i < z.length; i++) z[i] += (rand() - 0.5) * 0.0016;
    return { values: z, label: `demo ${k + 1}`, real: false };
  });
}

export function Calibrate() {
  const twin = useTwin();
  const online = twin.status?.bundle === true;
  const [cls, setCls] = useState<Cls>("V");
  const [batch, setBatch] = useState<Target[]>([]);
  const [pick, setPick] = useState(0);
  const [seed, setSeed] = useState(0);
  const [err, setErr] = useState<string | null>(null);
  const [imported, setImported] = useState<Target | null>(null);
  const [readingFile, setReadingFile] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const [fileWarning, setFileWarning] = useState<string | null>(null);
  const [, setTick] = useState(0);
  const cal = useRef<Calibrator | null>(null);
  const [running, setRunning] = useState(false);
  const [fit, setFit] = useState<ReturnType<Calibrator["best"]> | null>(null);
  const [whatIf, setWhatIf] = useState<Params | null>(null);
  const [active, setActive] = useState<string[]>([]);
  const [serverResult, setServer] = useState<{ target: Target; r2: number; z: number[]; ms: number } | null>(null);
  const [serverBusy, setServerBusy] = useState(false);

  // load targets
  useEffect(() => {
    let alive = true;
    setErr(null);
    if (!online) {
      setBatch(demoTargets(cls, 12));
      setPick(0);
      return;
    }
    backend
      .real({ beat_type: cls, n: 12, random_state: seed })
      .then((r: BeatsResponse) => {
        if (!alive) return;
        setBatch(r.values.map((v, i) => ({ values: Float64Array.from(v), label: `rec ${r.info.records?.[i] ?? "?"}`, real: true })));
        setPick(0);
      })
      .catch((e) => alive && setErr(String(e.message ?? e)));
    return () => {
      alive = false;
    };
  }, [cls, seed, online]);

  const target = imported ?? batch[pick];
  const targetLabel = imported ? "imported ECG" : target?.real ? "MIT-BIH beat" : "demo target";
  const server = serverResult?.target === target ? serverResult : null;

  const loadFile = async (file: File) => {
    setFileError(null);
    setFileWarning(null);
    setReadingFile(true);
    try {
      if (file.size > MAX_BEAT_FILE_BYTES) throw new Error("Choose a CSV smaller than 16 KB.");
      const values = importBeat(await file.text());
      setImported({ values, label: file.name, real: false });
      setFileWarning(alignmentWarning(values));
    } catch (e) {
      setFileError((e as Error).message);
    } finally {
      setReadingFile(false);
    }
  };

  // reset the fit when the target changes
  useEffect(() => {
    cal.current = null;
    setFit(null);
    setWhatIf(null);
    setActive([]);
    setServer(null);
    setRunning(false);
  }, [target]);

  // animate the fit
  useEffect(() => {
    if (!running) return;
    let raf = 0;
    const step = () => {
      const c = cal.current;
      if (!c) return;
      c.advance(4);
      setTick((t) => t + 1);
      if (c.done) {
        const best = c.best();
        setFit(best);
        setWhatIf(cloneParams(best.params));
        setActive([]);
        setRunning(false);
        return;
      }
      raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
  }, [running]);

  const start = () => {
    if (!target) return;
    cal.current = new Calibrator(target.values, twin.space, 300, 0.03);
    setFit(null);
    setWhatIf(null);
    setActive([]);
    setServer(null);
    setRunning(true);
  };

  const verify = async () => {
    if (!target || imported) return;
    setServerBusy(true);
    const t0 = performance.now();
    try {
      const r = await backend.calibrate(Array.from(target.values), 300);
      setServer({ target, r2: r.r2, z: r.fit.values[0], ms: performance.now() - t0 });
    } catch (e) {
      setErr(String((e as Error).message ?? e));
    } finally {
      setServerBusy(false);
    }
  };

  const c = cal.current;
  const live = c ? c.best() : null;
  const shown = fit ?? live;

  const residual = useMemo(() => {
    if (!shown || !target) return null;
    return Float64Array.from(shown.z, (v, i) => v - target.values[i]);
  }, [shown, target]);

  const scenarioZ = useMemo(() => (whatIf && fit ? simulate(whatIf, fit.z0) : null), [whatIf, fit]);
  const scenarioDelta = useMemo(() => (scenarioZ && fit ? Float64Array.from(scenarioZ, (v, i) => v - fit.z[i]) : null), [scenarioZ, fit]);

  const toggle = (id: string) => {
    if (!fit) return;
    const next = active.includes(id) ? active.filter((a) => a !== id) : [...active, id];
    const p = cloneParams(fit.params);
    for (const s of SCENARIOS) if (next.includes(s.id)) s.apply(p);
    setActive(next);
    setWhatIf(p);
  };

  const editWhatIf = (w: Wave, f: Field, v: number) =>
    setWhatIf((p) => {
      const n = cloneParams(p!);
      n[w][f] = v;
      return n;
    });

  const steps = c?.steps ?? 300;
  const lossX = useMemo(() => Float64Array.from({ length: steps + 1 }, (_, i) => i), [steps]);

  return (
    <div className="module">
      <div className="module-head">
        <h1>Calibrate and what-if</h1>
        <p>
          Fit the twin to one beat: the simulator's 15 parameters and starting value are tuned until its beat matches. Then change the
          fitted twin and see what the beat would look like.
        </p>
        {!imported && (
          <div className="tools">
            <Seg value={cls} options={["S", "V", "F"] as const} onChange={setCls} />
            {online && (
              <button className="btn ghost" onClick={() => setSeed((s) => s + 1)}>
                New sample
              </button>
            )}
          </div>
        )}
      </div>

      <Steps items={["Pick a target beat or import a CSV", "Press Calibrate twin and watch it fit", "Try a what-if or move a slider"]} />

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 2.2fr)" }} data-stack>
        <Panel
          code="A"
          title="Target beat"
          right={<span className="tag">{targetLabel}</span>}
        >
          {err && <div className="note" style={{ color: C.alert, marginBottom: 8 }}>{err}</div>}
          <label className="field-label" htmlFor="ecg-file">Import ECG beat (CSV)</label>
          <input
            id="ecg-file"
            className="ecg-file"
            type="file"
            accept=".csv,text/csv"
            disabled={readingFile}
            aria-describedby="ecg-file-format"
            aria-invalid={fileError ? true : undefined}
            onChange={(event) => {
              const file = event.currentTarget.files?.[0];
              event.currentTarget.value = "";
              if (file) void loadFile(file);
            }}
          />
          <p className="note ecg-file-note" id="ecg-file-format">
            One 360 Hz beat, 216 rows in mV, with the R peak at row 73. Optional header: amplitude_mv.
            The browser scales the amplitude for fitting and keeps the file local.
          </p>
          {fileError && <p className="note ecg-file-error" role="alert">{fileError}</p>}
          {imported ? (
            <>
              <p className="note" role="status">Selected: {imported.label}</p>
              {fileWarning && <p className="note ecg-file-warning" role="status">{fileWarning}</p>}
              <button className="btn ghost" disabled={readingFile} onClick={() => { setImported(null); setFileError(null); setFileWarning(null); }}>
                Use {online ? "MIT-BIH" : "demo"} examples
              </button>
            </>
          ) : (
            <>
              <div className="thumbs">
                {batch.map((t, i) => (
                  <button key={i} className={`thumb ${i === pick ? "on" : ""}`} onClick={() => setPick(i)}>
                    <TracePlot series={[{ y: t.values, color: i === pick ? C.real : C.dim, width: 1.1 }]} x={TIME_MS} height={40} bare tooltip={false} paper={false} yTicks={2} />
                    <div className="thumb-k">
                      <span>{t.label}</span>
                      <span>{cls}</span>
                    </div>
                  </button>
                ))}
              </div>
              <div className="note" style={{ marginTop: 10 }}>
                {online ? (
                  <>
                    Beats from MIT-BIH <b>validation patients</b>, never used to train the generators. {twin.status?.attribution}
                  </>
                ) : (
                  <>
                    <b>Server offline:</b> these demo targets are simulated beats with random changes and noise. Connect the server
                    with a model bundle for MIT-BIH examples.
                  </>
                )}
              </div>
            </>
          )}
          <div style={{ display: "flex", gap: 8, marginTop: 12, flexWrap: "wrap" }}>
            <button className="btn primary" onClick={start} disabled={!target || running || readingFile}>
              {running ? "Fitting…" : fit ? "Re-run fit" : "Calibrate twin"}
            </button>
            {online && !imported && (
              <button className="btn ghost" onClick={verify} disabled={!fit || serverBusy} title="Run simheart.api.calibrate (PyTorch) on the server">
                {serverBusy ? "Running…" : "Check on server"}
              </button>
            )}
          </div>
          <div style={{ marginTop: 10 }}>
            <Progress value={c ? c.step / c.steps : 0} />
            <div className="hint" style={{ marginTop: 6, display: "flex", justifyContent: "space-between" }}>
              <span>
                step {c?.step ?? 0} / {steps}
              </span>
              <span>Adam · 5 starts · cosine lr</span>
            </div>
          </div>
        </Panel>

        <Panel
          code="B"
          title="Fit"
          right={<Legend items={[{ color: C.real, label: targetLabel }, { color: C.twin, label: "twin fit" }, ...(server ? [{ color: C.simgan, label: "server fit", dash: true }] : [])]} />}
          flush
        >
          <div style={{ padding: "6px 6px 0" }}>
            <TracePlot
              x={TIME_MS}
              height={250}
              series={[
                ...(target ? [{ y: target.values, color: C.real, width: 1.6, label: targetLabel }] : []),
                ...(server ? [{ y: server.z, color: C.simgan, width: 1.3, dash: [5, 4], label: "server" }] : []),
                ...(shown ? [{ y: shown.z, color: C.twin, width: 1.8, glow: true, label: "twin" }] : []),
              ]}
            />
            <TracePlot
              x={TIME_MS}
              height={90}
              yTicks={2}
              yDomain={residual ? undefined : [-0.002, 0.002]}
              series={residual ? [{ y: residual, color: C.alert, width: 1.2, label: "residual" }] : []}
              xLabel={residual ? "residual = twin − target" : "residual · appears after the fit"}
            />
          </div>
          <div className="readouts">
            <Readout k="R²" help="How much of the beat's shape the twin explains: 1 is a perfect match" v={shown ? shown.r2.toFixed(4) : "—"} color={shown && shown.r2 > 0.9 ? C.twin : undefined} sub="share of variance explained" />
            <Readout k="SSE" help="Sum of squared differences between the twin and the target: lower is better" v={shown ? shown.loss.toExponential(2) : "—"} sub="sum of squared error" />
            <Readout k="Best start" help="The five fits start from different typical beats; this one ended closest" v={shown ? shown.name : "—"} sub="of N · S · V · F · McSharry" />
            {!imported && (
              <Readout
                k="Server check"
                help="The same fit run by the Python model on the server, to confirm the browser result"
                v={server ? server.r2.toFixed(4) : "—"}
                sub={server ? `R², PyTorch · ${(server.ms / 1000).toFixed(1)} s` : online ? "after fitting" : "needs server"}
              />
            )}
          </div>
        </Panel>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) minmax(0, 2.2fr)" }} data-stack>
        <Panel code="C" title="Convergence" right={<span className="hint">log₁₀ SSE per start</span>} flush>
          <div style={{ padding: "6px 6px 0" }}>
            <TracePlot
              x={lossX}
              xDomain={[0, steps]}
              xStep={50}
              paper={false}
              xLabel="step"
              height={210}
              fmtY={(v) => v.toFixed(1)}
              fmtX={(v) => `${v}`}
              series={(c?.runs ?? []).map((r, i) => ({
                y: r.history.map((h) => Math.log10(Math.max(h, 1e-12))),
                x: lossX,
                color: START_COLOR[i],
                width: live && live.name === r.name ? 2 : 1,
                alpha: live && live.name === r.name ? 1 : 0.55,
                label: r.name,
              }))}
              yDomain={c ? undefined : [-5, -1]}
            />
          </div>
          <div className="panel-body">
            <Legend items={START_NAMES.map((n, i) => ({ color: START_COLOR[i], label: n }))} />
            <div className="note" style={{ marginTop: 8 }}>
              Five fits run side by side from different starting beats; the one with the lowest error is kept, as in{" "}
              <code>simheart.simulator.fitting</code>. Gradients are exact (worked out by hand), so this runs in the browser.
            </div>
          </div>
        </Panel>

        <Panel code="D" title="What-if on the calibrated twin" right={<span className="tag">model what-if · not a prediction</span>} flush>
          {!fit || !whatIf || !scenarioZ || !scenarioDelta ? (
            <div className="empty">
              <h3>Calibrate first</h3>
              <div className="note">After the fit, apply scenarios or move any parameter to see how the twin's beat would change.</div>
            </div>
          ) : (
            <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.5fr) minmax(250px, 1fr)", gap: 0 }} data-stack>
              <div style={{ borderRight: "1px solid var(--line)", minWidth: 0 }}>
                <div className="panel-body" style={{ display: "flex", gap: 6, flexWrap: "wrap", borderBottom: "1px solid var(--line)" }}>
                  {SCENARIOS.map((s) => (
                    <button key={s.id} className={`chip ${active.includes(s.id) ? "on" : ""}`} onClick={() => toggle(s.id)}>
                      {s.label}
                    </button>
                  ))}
                  <button className="chip" onClick={() => { setActive([]); setWhatIf(cloneParams(fit.params)); }}>
                    clear
                  </button>
                </div>
                <div style={{ padding: "6px 6px 0" }}>
                  <Legend items={[
                    { color: C.real, label: targetLabel },
                    { color: C.twin, label: "calibrated fit", dash: true },
                    { color: C.text, label: "synthetic what-if" },
                  ]} />
                  <TracePlot
                    x={TIME_MS}
                    height={230}
                    series={[
                      ...(target ? [{ y: target.values, color: C.real, width: 1.2, label: targetLabel }] : []),
                      { y: fit.z, color: C.twin, width: 1.4, dash: [4, 3], label: "calibrated" },
                      { y: scenarioZ, color: C.text, width: 1.9, glow: true, label: "synthetic what-if" },
                    ]}
                  />
                  <TracePlot
                    x={TIME_MS}
                    height={86}
                    yTicks={2}
                    bands={[{ lo: new Float64Array(scenarioDelta.length), hi: scenarioDelta, color: C.alert, alpha: 0.25 }]}
                    series={[{ y: scenarioDelta, color: C.alert, width: 1.2, label: "Δ" }]}
                    xLabel="Δ = what-if − calibrated"
                  />
                </div>
                <IntervalDelta base={fit.params} next={whatIf} />
                <p className="note panel-body">Intervals are model estimates. Amplitude is in simulator units. {NOTICE}</p>
              </div>
              <div className="panel-body" style={{ maxHeight: 520, overflowY: "auto" }}>
                <div className="field-label">Fine-tune (tick = calibrated value)</div>
                <WaveControls params={whatIf} reference={fit.params} onChange={editWhatIf} cls={imported ? undefined : cls} compact />
              </div>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

function IntervalDelta({ base, next }: { base: Params; next: Params }) {
  const a = modelIntervals(base);
  const b = modelIntervals(next);
  const row = (k: keyof typeof a, label: string) => {
    const d = b[k] - a[k];
    return <Readout k={label} v={Math.round(b[k])} unit="ms" sub={Math.abs(d) < 0.5 ? "unchanged" : `${d > 0 ? "+" : ""}${d.toFixed(0)} ms`} color={Math.abs(d) >= 0.5 ? C.text : undefined} />;
  };
  return (
    <div className="readouts">
      {row("pr", "PR (model)")}
      {row("qrs", "QRS (model)")}
      {row("qt", "QT (model)")}
    </div>
  );
}
