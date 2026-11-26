/**
 * Rhythm composer for the live monitor: strings simulated beats into a continuous strip.
 *
 * The simulator makes one 600 ms window per beat (−200 … +400 ms around the R-peak). The
 * composer places those windows on a timeline: a longer RR interval lengthens the flat
 * stretch between beats, a shorter one lets the next beat start sooner. Ectopic beats
 * follow textbook timing (display only, not part of the trained models):
 *
 *   V  premature (≈62 % of RR), full compensatory pause: the next sinus beat stays on time
 *   S  premature (≈72 % of RR), resets the sinus clock
 *   F  fusion: arrives about on time
 */

import { N_SAMPLES, R_PEAK_INDEX, SAMPLING_RATE, WAVES, cloneParams, simulate, type Params } from "./ecgsyn";

export type BeatClass = "N" | "S" | "V" | "F";

export interface RhythmConfig {
  hr: number; // sinus rate, beats per minute
  hrv: number; // beat-to-beat RR spread (fraction)
  ectopic: Exclude<BeatClass, "N">;
  burden: number; // share of beats that are ectopic (0 … 1)
  variability: number; // beat-to-beat morphology jitter (fraction)
}

export interface StreamBeat {
  cls: BeatClass;
  tR: number; // ms
  values: Float64Array;
  params: Params;
}

const PRE_MS = (R_PEAK_INDEX / SAMPLING_RATE) * 1000; // 200
const POST_MS = ((N_SAMPLES - R_PEAK_INDEX) / SAMPLING_RATE) * 1000; // 400

function gauss(rand: () => number) {
  const u = Math.max(rand(), 1e-12);
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * rand());
}

export function mulberry32(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export class RhythmStream {
  beats: StreamBeat[] = [];
  private sinusNext: number;
  private rand: () => number;
  private lastWasEctopic = false;

  constructor(
    public presets: Record<BeatClass, Params>,
    public config: RhythmConfig,
    seed = 7,
    startMs = 0,
  ) {
    this.rand = mulberry32(seed);
    this.sinusNext = startMs + 300;
  }

  private rr() {
    const base = 60000 / this.config.hr;
    return base * (1 + this.config.hrv * gauss(this.rand) * 0.5);
  }

  private makeBeat(cls: BeatClass, tR: number): StreamBeat {
    const p = cloneParams(this.presets[cls]);
    const v = this.config.variability;
    for (const w of WAVES) {
      p[w].a *= 1 + v * gauss(this.rand);
      p[w].theta_deg += v * 4 * gauss(this.rand);
    }
    return { cls, tR, values: simulate(p), params: p };
  }

  /** Schedule beats until the timeline reaches tMs (plus a look-ahead). */
  fill(tMs: number) {
    while (this.lastR() < tMs + 1500) {
      const { burden, ectopic } = this.config;
      const ectopicNow = !this.lastWasEctopic && this.rand() < burden;
      const rr = this.rr();
      const last = this.beats.at(-1);
      if (!ectopicNow || !last) {
        this.push(this.makeBeat("N", this.sinusNext));
        this.sinusNext += rr;
        this.lastWasEctopic = false;
        continue;
      }
      if (ectopic === "V") {
        const t = last.tR + 0.62 * rr;
        this.push(this.makeBeat("V", t));
        this.sinusNext += rr; // the blocked sinus beat is skipped: compensatory pause
      } else if (ectopic === "S") {
        const t = last.tR + 0.72 * rr;
        this.push(this.makeBeat("S", t));
        this.sinusNext = t + rr;
      } else {
        this.push(this.makeBeat("F", this.sinusNext - 0.03 * rr));
        this.sinusNext += rr;
      }
      this.lastWasEctopic = true;
    }
    // forget beats far in the past
    while (this.beats.length > 4 && this.beats[1].tR < tMs - 20000) this.beats.shift();
  }

  private push(b: StreamBeat) {
    this.beats.push(b);
  }

  private lastR() {
    return this.beats.length ? this.beats[this.beats.length - 1].tR : -Infinity;
  }

  /** The beat whose window covers t (the later beat wins where windows overlap). */
  beatAt(t: number): { beat: StreamBeat; index: number } | null {
    for (let i = this.beats.length - 1; i >= 0; i--) {
      const b = this.beats[i];
      if (t >= b.tR - PRE_MS && t < b.tR + POST_MS) {
        const idx = Math.round(((t - b.tR) / 1000) * SAMPLING_RATE) + R_PEAK_INDEX;
        return { beat: b, index: Math.max(0, Math.min(N_SAMPLES - 1, idx)) };
      }
    }
    return null;
  }

  /** Signal value at time t; flat stretches between windows are bridged linearly. */
  valueAt(t: number): number {
    const hit = this.beatAt(t);
    if (hit) return hit.beat.values[hit.index];
    let prev: StreamBeat | undefined;
    let next: StreamBeat | undefined;
    for (const b of this.beats) {
      if (b.tR + POST_MS <= t) prev = b;
      else if (b.tR - PRE_MS > t) {
        next = b;
        break;
      }
    }
    if (!prev || !next) return prev ? prev.values[N_SAMPLES - 1] : 0;
    const t0 = prev.tR + POST_MS;
    const t1 = next.tR - PRE_MS;
    const f = (t - t0) / Math.max(t1 - t0, 1);
    return prev.values[N_SAMPLES - 1] * (1 - f) + next.values[0] * f;
  }
}
