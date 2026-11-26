/**
 * ECGSYN (McSharry et al., 2003), a line-for-line port of src/simheart/simulator/ecgsyn.py.
 *
 * A point moves round the unit circle (one lap = one beat). As it passes the five wave
 * angles θ_P … θ_T, the ECG value z is pushed up or down:
 *
 *   dz/dt = −Σ_i a_i Δθ_i exp(−Δθ_i² / 2b_i²) − (z − z_base(t))
 *
 * Same time base as the Python model: 216 samples, h = 1/216, ω = 2π, R wave at sample 72,
 * baseline wander 0.005·sin(2π·0.25·t). Runs in the browser so sliders respond instantly.
 */

export const WAVES = ["P", "Q", "R", "S", "T"] as const;
export type Wave = (typeof WAVES)[number];
export type Field = "a" | "b" | "theta_deg";
export const FIELDS: readonly Field[] = ["a", "b", "theta_deg"];
export type WaveParams = Record<Field, number>;
export type Params = Record<Wave, WaveParams>;
export type Ranges = Record<Wave, Record<Field, [number, number]>>;

export const N_SAMPLES = 216;
export const SAMPLING_RATE = 360;
export const R_PEAK_INDEX = 72;
export const DT = 1 / 216;
export const OMEGA = 2 * Math.PI;
const BASELINE_AMPLITUDE = 0.005;
const RESPIRATION_HZ = 0.25;

/** Milliseconds from the R-peak at each sample: −200 … 397 ms. */
export const TIME_MS = Float64Array.from(
  { length: N_SAMPLES },
  (_, i) => ((i - R_PEAK_INDEX) / SAMPLING_RATE) * 1000,
);
/** One radian of the cycle in ms (the 216-sample window is one lap of 2π). */
export const MS_PER_RAD = (N_SAMPLES / SAMPLING_RATE) * 1000 / (2 * Math.PI);

/** The (x, y) path round the circle, integrated with Euler steps like the Python model. */
function phasePath() {
  const start = -OMEGA * DT * R_PEAK_INDEX;
  let x = Math.cos(start);
  let y = Math.sin(start);
  const theta = new Float64Array(N_SAMPLES);
  const xs = new Float64Array(N_SAMPLES);
  const ys = new Float64Array(N_SAMPLES);
  for (let i = 0; i < N_SAMPLES; i++) {
    theta[i] = Math.atan2(y, x);
    xs[i] = x;
    ys[i] = y;
    const alpha = 1 - Math.hypot(x, y);
    const nx = x + DT * (alpha * x - OMEGA * y);
    const ny = y + DT * (alpha * y + OMEGA * x);
    x = nx;
    y = ny;
  }
  return { theta, xs, ys };
}

export const PHASE = phasePath();
export const Z_BASE = Float64Array.from(
  { length: N_SAMPLES },
  (_, i) => BASELINE_AMPLITUDE * Math.sin(2 * Math.PI * RESPIRATION_HZ * i * DT),
);

const RAD = Math.PI / 180;

export function wrapAngle(d: number): number {
  return Math.atan2(Math.sin(d), Math.cos(d));
}

/** Each wave's push on dz/dt at every sample: [5][216]. */
export function waveDrives(params: Params): Float64Array[] {
  return WAVES.map((w) => {
    const { a, b, theta_deg } = params[w];
    const ti = theta_deg * RAD;
    const out = new Float64Array(N_SAMPLES);
    for (let i = 0; i < N_SAMPLES; i++) {
      const d = wrapAngle(PHASE.theta[i] - ti);
      out[i] = -a * d * Math.exp(-(d * d) / (2 * b * b));
    }
    return out;
  });
}

/** Euler-integrate z for a given total drive. */
export function integrate(drive: ArrayLike<number>, z0 = 0): Float64Array {
  const z = new Float64Array(N_SAMPLES);
  let v = z0;
  z[0] = v;
  for (let k = 0; k < N_SAMPLES - 1; k++) {
    v = v + DT * (drive[k] - (v - Z_BASE[k]));
    z[k + 1] = v;
  }
  return z;
}

/** The simulated beat (simulator units, same scale as the real and generated beats). */
export function simulate(params: Params, z0 = 0): Float64Array {
  const drives = waveDrives(params);
  const total = new Float64Array(N_SAMPLES);
  for (const d of drives) for (let i = 0; i < N_SAMPLES; i++) total[i] += d[i];
  return integrate(total, z0);
}

/**
 * The beat split into each wave's own contribution. z is linear in the drive, so the
 * waves' separately integrated responses (z0 = 0, no baseline) add up to the beat minus
 * the baseline/z0 response.
 */
export function decompose(params: Params): Float64Array[] {
  return waveDrives(params).map((drive) => {
    const z = new Float64Array(N_SAMPLES);
    let v = 0;
    for (let k = 0; k < N_SAMPLES - 1; k++) {
      v = v + DT * (drive[k] - v);
      z[k + 1] = v;
    }
    return z;
  });
}

export function makeParams(
  a: number[],
  b: number[],
  thetaDeg: number[],
): Params {
  const out = {} as Params;
  WAVES.forEach((w, i) => (out[w] = { a: a[i], b: b[i], theta_deg: thetaDeg[i] }));
  return out;
}

export function cloneParams(p: Params): Params {
  const out = {} as Params;
  for (const w of WAVES) out[w] = { ...p[w] };
  return out;
}

/** Normal beat, McSharry et al. (2003), Table 1. */
export const MCSHARRY = makeParams(
  [1.2, -5.0, 30.0, -7.5, 0.75],
  [0.25, 0.1, 0.1, 0.1, 0.4],
  [-70, -15, 0, 15, 100],
);

/** Per-class starting values from the SimGANs authors' code (as in ecgsyn.TYPICAL_PARAMS). */
export const TYPICAL: Record<"N" | "S" | "V" | "F", Params> = {
  N: makeParams([0.7, -7.0, 30.0, -3.0, 0.2], [0.25, 0.1, 0.1, 0.1, 0.4], [-90, -15, 0, 15, 160]),
  S: makeParams([0.2, -1.0, 30.0, -10.0, 0.2], [0.25, 0.1, 0.1, 0.1, 0.4], [-90, -15, 0, 15, 160]),
  V: makeParams([0.1, 0.0, 30.0, -10.0, 0.5], [0.6, 0.1, 0.1, 0.1, 0.2], [-90, -15, 0, 15, 160]),
  F: makeParams([0.8, -10.0, 30.0, -10.0, 0.5], [0.25, 0.1, 0.1, 0.1, 0.2], [-90, -15, 0.03, 15, 160]),
};

/** Wide slider limits used when no model bundle is connected. */
export const DEFAULT_RANGES: Ranges = {
  P: { a: [-1.5, 2.5], b: [0.05, 0.9], theta_deg: [-140, -40] },
  Q: { a: [-25, 5], b: [0.03, 0.4], theta_deg: [-50, -3] },
  R: { a: [0, 60], b: [0.03, 0.4], theta_deg: [-20, 20] },
  S: { a: [-30, 5], b: [0.03, 0.4], theta_deg: [3, 55] },
  T: { a: [-2, 3], b: [0.08, 0.9], theta_deg: [60, 210] },
};

/**
 * Model intervals read off the parameters: wave centres and ±2b extents converted to ms.
 * They describe the simulated beat, not a clinical measurement.
 */
export function modelIntervals(p: Params) {
  const t = (w: Wave) => p[w].theta_deg * RAD * MS_PER_RAD;
  const half = (w: Wave) => 2 * p[w].b * MS_PER_RAD;
  const qrsOn = t("Q") - half("Q");
  const qrsOff = t("S") + half("S");
  return {
    pr: t("R") - t("P"),
    qrs: qrsOff - qrsOn,
    qt: t("T") + half("T") - qrsOn,
  };
}

export function stats(z: ArrayLike<number>) {
  let min = Infinity;
  let max = -Infinity;
  let iMax = 0;
  for (let i = 0; i < z.length; i++) {
    if (z[i] < min) min = z[i];
    if (z[i] > max) {
      max = z[i];
      iMax = i;
    }
  }
  return { min, max, iMax, span: max - min };
}
