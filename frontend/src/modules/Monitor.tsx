import { useEffect, useMemo, useRef, useState } from "react";
import { useTwin } from "../backend";
import { TIME_MS, modelIntervals, simulate } from "../engine/ecgsyn";
import { RhythmStream, type BeatClass, type RhythmConfig } from "../engine/rhythm";
import { C, TracePlot } from "../ui/Plot";
import { Panel, Readout, Seg, Slider } from "../ui/controls";
import { PhasePortrait, type PortraitState } from "../ui/PhasePortrait";

export const CLASS_COLOR: Record<BeatClass, string> = { N: C.dim, S: C.wave.S, V: C.alert, F: C.real };
export const CLASS_NAME: Record<BeatClass, string> = {
  N: "Normal",
  S: "Supraventricular ectopic",
  V: "Ventricular ectopic",
  F: "Fusion",
};

interface Vitals {
  hr: number;
  rr: number;
  counts: Record<BeatClass, number>;
  last: BeatClass;
  pvcPerMin: number;
}

export function Monitor() {
  const twin = useTwin();
  const [config, setConfig] = useState<RhythmConfig>({ hr: 72, hrv: 0.06, ectopic: "V", burden: 0.12, variability: 0.03 });
  const [speed, setSpeed] = useState<25 | 50>(25);
  const [frozen, setFrozen] = useState(false);
  const [vitals, setVitals] = useState<Vitals>({ hr: 0, rr: 0, counts: { N: 0, S: 0, V: 0, F: 0 }, last: "N", pvcPerMin: 0 });

  const stream = useMemo(() => new RhythmStream(twin.presets, config), []);
  stream.config = config;
  stream.presets = twin.presets;

  const portrait = useRef<PortraitState>({ values: simulate(twin.presets.N), index: null, params: twin.presets.N, color: C.twin });
  const canvas = useRef<HTMLCanvasElement>(null);
  const frozenRef = useRef(frozen);
  frozenRef.current = frozen;
  const speedRef = useRef(speed);
  speedRef.current = speed;

  useEffect(() => {
    let raf = 0;
    let clock = 0; // ms of simulated time
    let lastFrame = performance.now();
    let seenBeats = new Set<number>();
    const counts: Record<BeatClass, number> = { N: 0, S: 0, V: 0, F: 0 };
    const rrs: number[] = [];
    const ectopicTimes: number[] = [];
    let lastR: number | null = null;
    const cv = canvas.current!;
    const height = 260;

    const frame = (now: number) => {
      raf = requestAnimationFrame(frame);
      const dt = Math.min(now - lastFrame, 100);
      lastFrame = now;
      if (!frozenRef.current) clock += dt;
      stream.fill(clock);

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

      // paper: 25 mm/s → 1 mm = 40 ms. A minor box is 1 mm.
      const pxPerMm = 4.2;
      const msPerPx = 1000 / (speedRef.current * pxPerMm);
      const windowMs = width * msPerPx;
      g.lineWidth = 1;
      for (let mm = 0; mm * pxPerMm < Math.max(width, height); mm++) {
        const p = Math.round(mm * pxPerMm) + 0.5;
        g.strokeStyle = mm % 5 === 0 ? "#1a2329" : "#10171c";
        g.beginPath();
        if (p < width) {
          g.moveTo(p, 0);
          g.lineTo(p, height);
        }
        if (p < height) {
          g.moveTo(0, p);
          g.lineTo(width, p);
        }
        g.stroke();
      }

      const cursorPx = (clock % windowMs) / msPerPx;
      const gapPx = 28;
      const baseY = height * 0.62;
      const yScale = (height * 0.52) / 0.06; // sim units → px
      const timeAt = (px: number) => (px <= cursorPx ? clock - (cursorPx - px) * msPerPx : clock - (cursorPx - px + width) * msPerPx);

      // trace (two sub-samples per pixel so R peaks don't alias away)
      g.strokeStyle = C.twin;
      g.lineWidth = 1.6;
      g.shadowColor = C.twin;
      g.shadowBlur = 6;
      g.lineJoin = "round";
      const drawRun = (from: number, to: number) => {
        if (to <= from) return;
        g.beginPath();
        for (let px = from; px <= to; px += 0.5) {
          const y = baseY - stream.valueAt(timeAt(px)) * yScale;
          if (px === from) g.moveTo(px, y);
          else g.lineTo(px, y);
        }
        g.stroke();
      };
      drawRun(0, Math.floor(cursorPx));
      drawRun(Math.ceil(cursorPx + gapPx), width);
      g.shadowBlur = 0;

      // older sweep is slightly dimmer
      g.fillStyle = "rgba(7,9,11,0.28)";
      g.fillRect(cursorPx + gapPx, 0, width - cursorPx - gapPx, height);

      // cursor
      const grad = g.createLinearGradient(cursorPx - 40, 0, cursorPx, 0);
      grad.addColorStop(0, "rgba(79,227,193,0)");
      grad.addColorStop(1, "rgba(79,227,193,0.10)");
      g.fillStyle = grad;
      g.fillRect(cursorPx - 40, 0, 40, height);
      g.fillStyle = C.twin;
      g.beginPath();
      g.arc(cursorPx, baseY - stream.valueAt(clock) * yScale, 3, 0, Math.PI * 2);
      g.fill();

      // beat annotations
      g.font = "600 10px 'IBM Plex Mono', monospace";
      g.textAlign = "center";
      let prev: number | null = null;
      for (const b of stream.beats) {
        if (b.tR > clock || b.tR < clock - windowMs + gapPx * msPerPx) {
          prev = b.tR;
          continue;
        }
        const px = b.tR <= clock && clock - b.tR <= cursorPx * msPerPx ? cursorPx - (clock - b.tR) / msPerPx : cursorPx - (clock - b.tR) / msPerPx + width;
        if (px > cursorPx && px < cursorPx + gapPx) continue;
        g.fillStyle = CLASS_COLOR[b.cls];
        g.fillText(b.cls, px, 16);
        if (b.cls !== "N") {
          g.strokeStyle = CLASS_COLOR[b.cls];
          g.globalAlpha = 0.35;
          g.strokeRect(px - 9, 5, 18, 16);
          g.globalAlpha = 1;
        }
        if (prev !== null) {
          g.fillStyle = "#5d6b73";
          g.font = "10px 'IBM Plex Mono', monospace";
          g.fillText(`${Math.round(b.tR - prev)}`, px - (b.tR - prev) / msPerPx / 2, height - 10);
          g.font = "600 10px 'IBM Plex Mono', monospace";
        }
        prev = b.tR;
      }

      // vitals: register beats as the cursor passes their R-peak
      for (const b of stream.beats) {
        if (b.tR > clock || seenBeats.has(b.tR)) continue;
        seenBeats.add(b.tR);
        counts[b.cls]++;
        if (lastR !== null) rrs.push(b.tR - lastR);
        if (rrs.length > 8) rrs.shift();
        lastR = b.tR;
        if (b.cls !== "N") ectopicTimes.push(b.tR);
        while (ectopicTimes.length && ectopicTimes[0] < clock - 60000) ectopicTimes.shift();
        const meanRR = rrs.length ? rrs.reduce((s, v) => s + v, 0) / rrs.length : 0;
        setVitals({
          hr: meanRR ? 60000 / meanRR : 0,
          rr: rrs.at(-1) ?? 0,
          counts: { ...counts },
          last: b.cls,
          pvcPerMin: ectopicTimes.length * (60000 / Math.min(Math.max(clock, 1), 60000)),
        });
      }
      if (seenBeats.size > 200) seenBeats = new Set([...seenBeats].slice(-50));

      const hit = stream.beatAt(clock);
      if (hit) {
        portrait.current = {
          values: hit.beat.values,
          index: hit.index,
          params: hit.beat.params,
          color: hit.beat.cls === "N" ? C.twin : CLASS_COLOR[hit.beat.cls],
        };
      }
    };
    raf = requestAnimationFrame(frame);
    return () => cancelAnimationFrame(raf);
  }, [stream]);

  const total = vitals.counts.N + vitals.counts.S + vitals.counts.V + vitals.counts.F;
  const set = <K extends keyof RhythmConfig>(k: K, v: RhythmConfig[K]) => setConfig((c) => ({ ...c, [k]: v }));

  return (
    <div className="module">
      <div className="module-head">
        <h1>
          <span>01</span>LIVE TWIN MONITOR
        </h1>
        <p>
          The heart model running continuously. Each beat is simulated by ECGSYN from its class parameters, then placed on a
          timeline by the rhythm composer.
        </p>
        <div className="tools">
          <Seg value={speed} options={[25, 50] as const} onChange={setSpeed} labels={{ 25: "25 mm/s", 50: "50 mm/s" }} />
          <button className="btn ghost" onClick={() => setFrozen((f) => !f)}>
            {frozen ? "▶ Resume" : "❚❚ Freeze"}
          </button>
        </div>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1fr) 260px" }} data-stack>
        <Panel
          code="A"
          title="Lead · simulated MLII-like strip"
          right={
            <>
              <span className="tag live">● live</span>
              <span className="tag syn">synthetic</span>
            </>
          }
          flush
        >
          <canvas ref={canvas} style={{ width: "100%", height: 260, display: "block" }} />
          <div className="readouts">
            <Readout k="Paper" v={`${speed}`} unit="mm/s" sub="1 box = 40 ms" />
            <Readout k="Last RR" v={vitals.rr ? Math.round(vitals.rr) : "—"} unit="ms" />
            <Readout k="Beats" v={total} sub={`N ${vitals.counts.N} · S ${vitals.counts.S} · V ${vitals.counts.V} · F ${vitals.counts.F}`} />
            <Readout k="Last beat" v={vitals.last} color={CLASS_COLOR[vitals.last]} sub={CLASS_NAME[vitals.last]} />
          </div>
        </Panel>

        <Panel code="B" title="Vitals" flush>
          <div className="vitals">
          <div className="vital">
            <div className="vital-k">
              <span>Heart rate</span>
              <span>bpm</span>
            </div>
            <div className="vital-v">{vitals.hr ? Math.round(vitals.hr) : "--"}</div>
          </div>
          <div className="vital">
            <div className="vital-k">
              <span>Ectopic</span>
              <span>/ min</span>
            </div>
            <div className="vital-v md" style={{ color: vitals.pvcPerMin > 6 ? C.alert : undefined }}>
              {vitals.pvcPerMin.toFixed(1)}
            </div>
          </div>
          <div className="vital">
            <div className="vital-k">
              <span>Burden (observed)</span>
              <span>%</span>
            </div>
            <div className="vital-v md">{total ? (((total - vitals.counts.N) / total) * 100).toFixed(1) : "0.0"}</div>
          </div>
          <div className="vital">
            <div className="note">
              Rates are counted from the strip like a bedside monitor (mean of the last 8 RR intervals). The ectopic rate is over the last
              minute of simulated time.
            </div>
          </div>
          </div>
        </Panel>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "minmax(0, 1.15fr) minmax(0, 1fr) minmax(0, 1fr)" }} data-stack>
        <Panel code="C" title="State space · (x, y, z)" right={<span className="hint">synced to cursor</span>} flush>
          <PhasePortrait source={portrait} height={318} />
          <div className="panel-body" style={{ borderTop: "1px solid var(--line)" }}>
            <div className="note">
              ECGSYN moves a point round the unit circle once per beat. Passing each wave angle pushes <b>z</b> up or down, which
              draws the wave. The curtain shows z lifted off the circle.
            </div>
          </div>
        </Panel>

        <Panel code="D" title="Rhythm composer">
          <div style={{ display: "grid", gap: 14 }}>
            <div>
              <div className="field-label">Sinus rate · bpm</div>
              <Slider k="HR" value={config.hr} min={40} max={150} step={1} digits={0} onChange={(v) => set("hr", Math.round(v))} reference={72} />
              <Slider k="HRV" value={config.hrv} min={0} max={0.2} step={0.005} onChange={(v) => set("hrv", v)} reference={0.06} />
            </div>
            <div>
              <div className="field-label" style={{ display: "flex", justifyContent: "space-between" }}>
                <span>Ectopic beat</span>
              </div>
              <Seg value={config.ectopic} options={["S", "V", "F"] as const} onChange={(v) => set("ectopic", v)} />
              <div style={{ height: 8 }} />
              <Slider k="%" value={config.burden * 100} min={0} max={40} step={1} digits={0} color={CLASS_COLOR[config.ectopic]} onChange={(v) => set("burden", v / 100)} reference={12} />
            </div>
            <div>
              <div className="field-label">Beat-to-beat morphology jitter</div>
              <Slider k="σ" value={config.variability} min={0} max={0.08} step={0.002} onChange={(v) => set("variability", v)} reference={0.03} />
            </div>
            <div className="note">
              <b>Timing rules (display only):</b> V arrives at ~62 % of RR with a full compensatory pause; S at ~72 % and resets the
              sinus clock; F fuses at about the normal time. The trained models make single beats, not rhythms.
            </div>
          </div>
        </Panel>

        <Panel code="E" title="Beat classes in the twin" right={<span className="tag">{twin.presetSource === "bundle" ? "S/V/F: fitted medians" : "typical values"}</span>}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
            {(["N", "S", "V", "F"] as const).map((cls) => (
              <BeatCard key={cls} cls={cls} />
            ))}
          </div>
          <div className="note" style={{ marginTop: 10 }}>
            {twin.presetSource === "bundle"
              ? "S, V and F use the median of the parameters fitted to real training beats (from the model bundle). N uses the SimGANs authors' typical values."
              : "Using the SimGANs authors' typical values. Connect the server with a model bundle to use the medians fitted to MIT-BIH training beats."}
          </div>
        </Panel>
      </div>
    </div>
  );
}

function BeatCard({ cls }: { cls: BeatClass }) {
  const twin = useTwin();
  const p = twin.presets[cls];
  const z = useMemo(() => simulate(p), [p]);
  const iv = modelIntervals(p);
  return (
    <div className="wave-card" style={{ padding: 6 }}>
      <div className="thumb-k" style={{ marginBottom: 2 }}>
        <span style={{ color: CLASS_COLOR[cls], fontWeight: 600 }}>{cls}</span>
        <span>QRS {Math.round(iv.qrs)} ms</span>
      </div>
      <TracePlot series={[{ y: z, color: cls === "N" ? C.twin : CLASS_COLOR[cls], width: 1.4 }]} x={TIME_MS} height={64} bare tooltip={false} yDomain={[-0.025, 0.055]} />
    </div>
  );
}
