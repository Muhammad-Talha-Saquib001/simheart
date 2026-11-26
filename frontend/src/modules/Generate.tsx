import { useEffect, useMemo, useState } from "react";
import { backend, useTwin, type BeatsResponse } from "../backend";
import { TIME_MS } from "../engine/ecgsyn";
import { envelope, hfFraction, meanPairwise, medianOf, mmdRbf } from "../engine/metrics";
import { C, Legend, TracePlot } from "../ui/Plot";
import { Panel, Seg } from "../ui/controls";

type Cls = "S" | "V" | "F";
type Source = "real" | "dcgan" | "simdcgan";
const COLOR: Record<Source, string> = { real: C.real, dcgan: C.dcgan, simdcgan: C.simgan };
const TITLE: Record<Source, string> = { real: "Real · MIT-BIH validation", dcgan: "DCGAN", simdcgan: "SimDCGAN (physics-informed)" };

// SimGANs paper (Golany et al., ICML 2020), Table 1: precision at fixed recall, whole MIT-BIH database.
const PAPER_TABLE_1: Record<Cls, { recall: number; dcgan: number; simdcgan: number }> = {
  S: { recall: 0.41, dcgan: 0.58, simdcgan: 0.8 },
  V: { recall: 0.91, dcgan: 0.72, simdcgan: 0.84 },
  F: { recall: 0.6, dcgan: 0.2, simdcgan: 0.4 },
};

export function Generate() {
  const twin = useTwin();
  const status = twin.status;
  const online = status?.bundle === true;
  const [cls, setCls] = useState<Cls>("V");
  const [n, setN] = useState<16 | 32 | 64>(32);
  const [rs, setRs] = useState(0);
  const [seeds, setSeeds] = useState<Record<string, number | undefined>>({});
  const [data, setData] = useState<Partial<Record<Source, BeatsResponse>> & { real2?: BeatsResponse }>({});
  const [errors, setErrors] = useState<Partial<Record<Source, string>>>({});
  const [busy, setBusy] = useState(false);

  const seedOptions = (model: string) => (status?.models ?? []).filter((m) => m.model === model && m.beat_type === cls);

  useEffect(() => {
    if (!online) return;
    let alive = true;
    setBusy(true);
    const jobs: [Source | "real2", Promise<BeatsResponse>][] = [
      ["real", backend.real({ beat_type: cls, n, random_state: rs })],
      ["real2", backend.real({ beat_type: cls, n, random_state: rs + 1000 })],
      ["dcgan", backend.generate({ beat_type: cls, n, model: "dcgan", seed: seeds[`dcgan_${cls}`], random_state: rs })],
      ["simdcgan", backend.generate({ beat_type: cls, n, model: "simdcgan", seed: seeds[`simdcgan_${cls}`], random_state: rs })],
    ];
    Promise.allSettled(jobs.map(([, p]) => p)).then((res) => {
      if (!alive) return;
      const d: typeof data = {};
      const e: typeof errors = {};
      res.forEach((r, i) => {
        const k = jobs[i][0];
        if (r.status === "fulfilled") (d as Record<string, BeatsResponse>)[k] = r.value;
        else if (k !== "real2") e[k] = String(r.reason?.message ?? r.reason);
      });
      setData(d);
      setErrors(e);
      setBusy(false);
    });
    return () => {
      alive = false;
    };
  }, [online, cls, n, rs, seeds]);

  const metrics = useMemo(() => {
    const real = data.real?.values;
    if (!real) return null;
    const row = (beats?: number[][]) =>
      beats
        ? {
            mmd: mmdRbf(beats, real),
            div: meanPairwise(beats),
            hf: medianOf(beats.map((b) => hfFraction(b))),
          }
        : null;
    return {
      realFloor: data.real2 ? mmdRbf(data.real2.values, real) : null,
      real: { mmd: null as number | null, div: meanPairwise(real), hf: medianOf(real.map((b) => hfFraction(b))) },
      dcgan: row(data.dcgan?.values),
      simdcgan: row(data.simdcgan?.values),
    };
  }, [data]);

  const envs = useMemo(() => {
    const out: Partial<Record<Source, ReturnType<typeof envelope>>> = {};
    for (const s of ["real", "dcgan", "simdcgan"] as const) if (data[s]?.values.length) out[s] = envelope(data[s]!.values);
    return out;
  }, [data]);

  if (!online) return <Offline linkOnline={twin.link === "online"} error={status?.bundle_error ?? null} />;

  return (
    <div className="module">
      <div className="module-head">
        <h1>
          <span>04</span>GENERATOR LAB
        </h1>
        <p>
          Beats drawn from the trained generators next to real beats of the same class. SimDCGAN was trained with the Euler loss, which
          pulls its beats towards the heart equations; DCGAN was not.
        </p>
        <div className="tools">
          <Seg value={cls} options={["S", "V", "F"] as const} onChange={setCls} />
          <Seg value={n} options={[16, 32, 64] as const} onChange={setN} labels={{ 16: "n 16", 32: "n 32", 64: "n 64" }} />
          <button className="btn primary" onClick={() => setRs((r) => r + 1)} disabled={busy}>
            {busy ? "Sampling…" : "Resample"}
          </button>
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "repeat(3, minmax(0, 1fr))" }} data-stack>
        {(["real", "dcgan", "simdcgan"] as const).map((s) => {
          const d = data[s];
          const e = envs[s];
          const opts = s === "real" ? [] : seedOptions(s);
          return (
            <Panel
              key={s}
              code={s === "real" ? "A" : s === "dcgan" ? "B" : "C"}
              title={TITLE[s]}
              right={
                <>
                  {opts.length > 1 && (
                    <select
                      className="seg"
                      style={{ padding: "3px 6px", fontFamily: "var(--mono)", fontSize: 11 }}
                      value={seeds[`${s}_${cls}`] ?? opts.find((o) => o.is_default)?.seed ?? opts[0].seed}
                      onChange={(ev) => setSeeds((m) => ({ ...m, [`${s}_${cls}`]: +ev.target.value }))}
                    >
                      {opts.map((o) => (
                        <option key={o.seed} value={o.seed}>
                          seed {o.seed}
                          {o.is_default ? " · default" : ""}
                        </option>
                      ))}
                    </select>
                  )}
                  {s === "real" ? <span className="tag" style={{ color: C.real, borderColor: C.real }}>real</span> : <span className="tag syn">synthetic</span>}
                </>
              }
              flush
            >
              {errors[s] ? (
                <div className="empty">
                  <h3>Not available</h3>
                  <div className="note">{errors[s]}</div>
                </div>
              ) : d && e ? (
                <div style={{ padding: "6px 6px 0" }}>
                  <TracePlot
                    x={TIME_MS}
                    height={240}
                    yDomain={[-0.022, 0.05]}
                    bands={[{ lo: e.lo, hi: e.hi, color: COLOR[s], alpha: 0.1 }]}
                    series={[
                      ...d.values.map((v) => ({ y: v, color: COLOR[s], width: 0.8, alpha: 0.22 })),
                      { y: e.med, color: COLOR[s], width: 2, label: "median", glow: true },
                    ]}
                  />
                  <div className="hint" style={{ padding: "4px 8px 8px", display: "flex", justifyContent: "space-between" }}>
                    <span>{d.values.length} beats · band 5–95 %</span>
                    <span>{s === "real" ? `records ${[...new Set(d.info.records ?? [])].join(", ")}` : `seed ${d.info.seed}`}</span>
                  </div>
                </div>
              ) : (
                <div className="empty">
                  <div className="note">Sampling…</div>
                </div>
              )}
            </Panel>
          );
        })}
      </div>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.3fr) minmax(0, 1fr)" }} data-stack>
        <Panel
          code="D"
          title="Medians compared"
          right={<Legend items={(["real", "dcgan", "simdcgan"] as const).filter((s) => envs[s]).map((s) => ({ color: COLOR[s], label: s }))} />}
          flush
        >
          <div style={{ padding: "6px 6px 0" }}>
            <TracePlot
              x={TIME_MS}
              height={400}
              series={(["real", "dcgan", "simdcgan"] as const)
                .filter((s) => envs[s])
                .map((s) => ({ y: envs[s]!.med, color: COLOR[s], width: s === "real" ? 2.2 : 1.6, label: s }))}
            />
          </div>
        </Panel>

        <div className="grid" style={{ alignContent: "start" }}>
          <Panel code="E" title="Fidelity on this batch" right={<span className="hint">indicative · n = {n}</span>} flush>
            <table className="data">
              <thead>
                <tr>
                  <th>source</th>
                  <th className="num">MMD² vs real ↓</th>
                  <th className="num">diversity</th>
                  <th className="num">HF &gt;40 Hz ↓</th>
                </tr>
              </thead>
              <tbody>
                {metrics && (
                  <>
                    <tr>
                      <td style={{ color: C.real }}>real (other beats)</td>
                      <td className="num">{metrics.realFloor !== null ? metrics.realFloor.toFixed(4) : "—"}</td>
                      <td className="num">{(metrics.real.div * 1000).toFixed(2)}</td>
                      <td className="num">{(metrics.real.hf * 100).toFixed(2)} %</td>
                    </tr>
                    {(["dcgan", "simdcgan"] as const).map((s) => {
                      const m = metrics[s];
                      return (
                        <tr key={s}>
                          <td style={{ color: COLOR[s] }}>{s}</td>
                          <td className="num">{m ? m.mmd.toFixed(4) : "—"}</td>
                          <td className="num">{m ? (m.div * 1000).toFixed(2) : "—"}</td>
                          <td className="num">{m ? `${(m.hf * 100).toFixed(2)} %` : "—"}</td>
                        </tr>
                      );
                    })}
                  </>
                )}
              </tbody>
            </table>
            <div className="panel-body" style={{ borderTop: "1px solid var(--line)" }}>
              <div className="note">
                <b>MMD²</b>: how different the batch is from the real batch (the "real" row is a second real batch, i.e. the noise floor).
                <b> Diversity</b>: mean distance between two beats (×10⁻³). <b>HF</b>: median share of power above 40 Hz (jaggedness). Same
                definitions as <code>simheart.evaluation.metrics</code>; the report uses the full validation set.
              </div>
            </div>
          </Panel>

          <Panel code="F" title="Reference · SimGANs paper, Table 1" flush>
            <table className="data">
              <thead>
                <tr>
                  <th>class</th>
                  <th className="num">at recall</th>
                  <th className="num">DCGAN</th>
                  <th className="num">SimDCGAN</th>
                </tr>
              </thead>
              <tbody>
                {(["S", "V", "F"] as const).map((k) => {
                  const r = PAPER_TABLE_1[k];
                  return (
                    <tr key={k} style={{ background: k === cls ? "var(--panel-2)" : undefined }}>
                      <td>{k}</td>
                      <td className="num">{r.recall.toFixed(2)}</td>
                      <td className="num" style={{ color: C.dcgan }}>{r.dcgan.toFixed(2)}</td>
                      <td className="num" style={{ color: C.simgan }}>{r.simdcgan.toFixed(2)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
            <div className="panel-body" style={{ borderTop: "1px solid var(--line)" }}>
              <div className="note">
                Classifier precision at a fixed recall after augmentation, as published (Golany et al., ICML 2020). SimHeart's own results
                are in <code>outputs/report/results.md</code>.
              </div>
            </div>
          </Panel>
        </div>
      </div>
    </div>
  );
}

function Offline({ linkOnline, error }: { linkOnline: boolean; error: string | null }) {
  return (
    <div className="module">
      <div className="module-head">
        <h1>
          <span>04</span>GENERATOR LAB
        </h1>
        <p>Real beats next to DCGAN and SimDCGAN beats, with live fidelity measures.</p>
      </div>
      <Panel code="A" title={linkOnline ? "Model bundle not found" : "Server offline"}>
        <div className="empty">
          <h3>{linkOnline ? "Server is up, but it has no model bundle" : "Start the SimHeart server"}</h3>
          <div className="note" style={{ maxWidth: 560 }}>
            The generators run in PyTorch on the server. The monitor, morphology lab and calibration keep working in the browser
            meanwhile.
          </div>
          <div className="code">
            {linkOnline
              ? `# make the bundle (after training), then restart the server\n.venv/bin/python scripts/export_models.py\n# or point the server at a shared bundle\nSIMHEART_MODEL_DIR=path/to/simheart-models .venv/bin/uvicorn simheart.server:app`
              : `.venv/bin/pip install fastapi uvicorn\n.venv/bin/uvicorn simheart.server:app --port 8000`}
          </div>
          {error && <div className="hint" style={{ maxWidth: 640 }}>{error}</div>}
        </div>
      </Panel>
    </div>
  );
}
