/**
 * Sample-quality measures, ported from simheart.evaluation.metrics so the console can score
 * whatever batch is on screen. On a few dozen beats they are indicative only; the report's
 * numbers use the full validation set.
 */

type Beat = ArrayLike<number>;

function dist(a: Beat, b: Beat): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}

function median(values: number[]): number {
  const v = [...values].sort((p, q) => p - q);
  const m = v.length >> 1;
  return v.length % 2 ? v[m] : (v[m - 1] + v[m]) / 2;
}

/** Unbiased squared MMD with an RBF kernel; bandwidth = median pooled distance. */
export function mmdRbf(x: Beat[], y: Beat[]): number {
  const pooled = [...x, ...y];
  const all: number[] = [];
  for (let i = 0; i < pooled.length; i++)
    for (let j = 0; j < pooled.length; j++) {
      const d = dist(pooled[i], pooled[j]);
      if (d > 0) all.push(d);
    }
  const bw = median(all);
  const k = (d: number) => Math.exp(-(d * d) / (2 * bw * bw));
  let xx = 0;
  let yy = 0;
  let xy = 0;
  for (let i = 0; i < x.length; i++) for (let j = 0; j < x.length; j++) if (i !== j) xx += k(dist(x[i], x[j]));
  for (let i = 0; i < y.length; i++) for (let j = 0; j < y.length; j++) if (i !== j) yy += k(dist(y[i], y[j]));
  for (const a of x) for (const b of y) xy += k(dist(a, b));
  const n = x.length;
  const m = y.length;
  return xx / (n * (n - 1)) + yy / (m * (m - 1)) - (2 * xy) / (n * m);
}

/** Average distance between two different beats (0 = all identical). */
export function meanPairwise(x: Beat[]): number {
  let s = 0;
  for (let i = 0; i < x.length; i++) for (let j = 0; j < x.length; j++) if (i !== j) s += dist(x[i], x[j]);
  return s / (x.length * (x.length - 1));
}

/** Share of a beat's power above 40 Hz (mean removed). */
export function hfFraction(beat: Beat, fs = 360, cutoff = 40): number {
  const n = beat.length;
  let mean = 0;
  for (let i = 0; i < n; i++) mean += beat[i];
  mean /= n;
  let high = 0;
  let total = 0;
  for (let k = 1; k <= n >> 1; k++) {
    let re = 0;
    let im = 0;
    for (let i = 0; i < n; i++) {
      const ang = (-2 * Math.PI * k * i) / n;
      const v = beat[i] - mean;
      re += v * Math.cos(ang);
      im += v * Math.sin(ang);
    }
    const p = re * re + im * im;
    total += p;
    if ((k * fs) / n > cutoff) high += p;
  }
  return high / Math.max(total, 1e-24);
}

export function medianOf(values: number[]): number {
  return median(values);
}

/** Per-sample median and 5th–95th percentile band of a batch. */
export function envelope(beats: Beat[]) {
  const n = beats[0].length;
  const med = new Float64Array(n);
  const lo = new Float64Array(n);
  const hi = new Float64Array(n);
  const col: number[] = new Array(beats.length);
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < beats.length; j++) col[j] = beats[j][i];
    col.sort((p, q) => p - q);
    const q = (f: number) => {
      const pos = f * (col.length - 1);
      const l = Math.floor(pos);
      const h = Math.ceil(pos);
      return col[l] + (col[h] - col[l]) * (pos - l);
    };
    med[i] = q(0.5);
    lo[i] = q(0.05);
    hi[i] = q(0.95);
  }
  return { med, lo, hi };
}
