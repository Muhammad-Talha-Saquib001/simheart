/**
 * Calibration: fit the simulator's 15 parameters and z0 to one beat, in the browser.
 *
 * Mirrors simheart.simulator.fitting: the same constrained parameter space (widths stay
 * positive, waves stay in P < Q < R < S < T order, |θ_R| ≤ θ_R_max), Adam with cosine
 * learning-rate decay, summed squared error, several starting points with the best kept.
 *
 * The gradient is exact and written out by hand. z_{k+1} = (1 − h) z_k + h (drive_k + base_k)
 * is linear, so the adjoint is one backward pass: λ_k = 2e_k + (1 − h) λ_{k+1},
 * ∂L/∂drive_j = h λ_{j+1}, ∂L/∂z0 = λ_0.
 */

import {
  DT,
  MCSHARRY,
  N_SAMPLES,
  PHASE,
  TYPICAL,
  WAVES,
  Z_BASE,
  makeParams,
  type Params,
  wrapAngle,
} from "./ecgsyn";

const RAD = Math.PI / 180;
// Raw layout: [aP..aT, rbP..rbT, gP..gT] (rows a, b, θ; columns P..T), then z0 at index 15.
const A = 0;
const B = 5;
const G = 10;
const Z0 = 15;
const [P, Q, R, S, T] = [0, 1, 2, 3, 4];

export interface Space {
  b_min: number;
  theta_r_max: number; // radians
}
export const DEFAULT_SPACE: Space = { b_min: 0.02, theta_r_max: 30 * RAD };

const softplus = (x: number) => (x > 30 ? x : Math.log1p(Math.exp(x)));
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const invSoftplus = (y: number) => y + Math.log(-Math.expm1(-y));

/** Valid a, b, θ (radians) from a raw vector. */
export function toParams(raw: Float64Array, space: Space) {
  const a = Array.from({ length: 5 }, (_, i) => raw[A + i]);
  const b = Array.from({ length: 5 }, (_, i) => space.b_min + softplus(raw[B + i]));
  const g = (i: number) => raw[G + i];
  const tR = space.theta_r_max * Math.tanh(g(R));
  const tQ = tR - softplus(g(Q));
  const tP = tQ - softplus(g(P));
  const tS = tR + softplus(g(S));
  const tT = tS + softplus(g(T));
  return { a, b, theta: [tP, tQ, tR, tS, tT] };
}

export function rawToParams(raw: Float64Array, space: Space): Params {
  const { a, b, theta } = toParams(raw, space);
  return makeParams(a, b, theta.map((t) => t / RAD));
}

/** Inverse of toParams (needs b > b_min, ordered angles and |θ_R| < θ_R_max). */
export function toRaw(p: Params, z0: number, space: Space): Float64Array {
  const raw = new Float64Array(16);
  const th = WAVES.map((w) => p[w].theta_deg * RAD);
  WAVES.forEach((w, i) => {
    raw[A + i] = p[w].a;
    raw[B + i] = invSoftplus(Math.max(p[w].b - space.b_min, 1e-6));
  });
  const ratio = Math.max(-0.999, Math.min(0.999, th[R] / space.theta_r_max));
  raw[G + R] = Math.atanh(ratio);
  raw[G + Q] = invSoftplus(Math.max(th[R] - th[Q], 1e-6));
  raw[G + P] = invSoftplus(Math.max(th[Q] - th[P], 1e-6));
  raw[G + S] = invSoftplus(Math.max(th[S] - th[R], 1e-6));
  raw[G + T] = invSoftplus(Math.max(th[T] - th[S], 1e-6));
  raw[Z0] = z0;
  return raw;
}

/** Loss (sum of squared errors), the simulated beat and the gradient w.r.t. the raw vector. */
export function lossAndGrad(raw: Float64Array, target: ArrayLike<number>, space: Space) {
  const { a, b, theta } = toParams(raw, space);
  // forward: drive and its per-wave pieces needed for the gradient
  const delta = new Float64Array(5 * N_SAMPLES);
  const expo = new Float64Array(5 * N_SAMPLES);
  const drive = new Float64Array(N_SAMPLES);
  for (let w = 0; w < 5; w++) {
    const inv = 1 / (2 * b[w] * b[w]);
    for (let i = 0; i < N_SAMPLES; i++) {
      const d = wrapAngle(PHASE.theta[i] - theta[w]);
      const e = Math.exp(-d * d * inv);
      delta[w * N_SAMPLES + i] = d;
      expo[w * N_SAMPLES + i] = e;
      drive[i] -= a[w] * d * e;
    }
  }
  const z = new Float64Array(N_SAMPLES);
  let v = raw[Z0];
  z[0] = v;
  for (let k = 0; k < N_SAMPLES - 1; k++) {
    v = v + DT * (drive[k] - (v - Z_BASE[k]));
    z[k + 1] = v;
  }
  let loss = 0;
  const err = new Float64Array(N_SAMPLES);
  for (let k = 0; k < N_SAMPLES; k++) {
    err[k] = z[k] - target[k];
    loss += err[k] * err[k];
  }
  // backward through the recurrence
  const lam = new Float64Array(N_SAMPLES);
  lam[N_SAMPLES - 1] = 2 * err[N_SAMPLES - 1];
  for (let k = N_SAMPLES - 2; k >= 0; k--) lam[k] = 2 * err[k] + (1 - DT) * lam[k + 1];
  const gDrive = new Float64Array(N_SAMPLES);
  for (let j = 0; j < N_SAMPLES - 1; j++) gDrive[j] = DT * lam[j + 1];

  const grad = new Float64Array(16);
  const gTheta = new Float64Array(5);
  for (let w = 0; w < 5; w++) {
    const b2 = b[w] * b[w];
    let ga = 0;
    let gb = 0;
    let gt = 0;
    for (let i = 0; i < N_SAMPLES - 1; i++) {
      const d = delta[w * N_SAMPLES + i];
      const e = expo[w * N_SAMPLES + i];
      const g = gDrive[i];
      ga += g * -d * e;
      gb += g * -a[w] * d * e * ((d * d) / (b2 * b[w]));
      gt += g * a[w] * e * (1 - (d * d) / b2);
    }
    grad[A + w] = ga;
    grad[B + w] = gb * sigmoid(raw[B + w]);
    gTheta[w] = gt;
  }
  const sech2 = 1 - Math.tanh(raw[G + R]) ** 2;
  grad[G + R] = space.theta_r_max * sech2 * (gTheta[P] + gTheta[Q] + gTheta[R] + gTheta[S] + gTheta[T]);
  grad[G + Q] = -sigmoid(raw[G + Q]) * (gTheta[Q] + gTheta[P]);
  grad[G + P] = -sigmoid(raw[G + P]) * gTheta[P];
  grad[G + S] = sigmoid(raw[G + S]) * (gTheta[S] + gTheta[T]);
  grad[G + T] = sigmoid(raw[G + T]) * gTheta[T];
  grad[Z0] = lam[0];
  return { loss, z, grad };
}

export function r2(z: ArrayLike<number>, target: ArrayLike<number>): number {
  let mean = 0;
  for (let i = 0; i < target.length; i++) mean += target[i];
  mean /= target.length;
  let sse = 0;
  let sst = 0;
  for (let i = 0; i < target.length; i++) {
    sse += (z[i] - target[i]) ** 2;
    sst += (target[i] - mean) ** 2;
  }
  return 1 - sse / sst;
}

export const START_NAMES = ["N", "S", "V", "F", "McSharry"] as const;

/** One Adam run per starting point, stepped a few iterations at a time (for animation). */
export class Calibrator {
  readonly steps: number;
  readonly lr: number;
  step = 0;
  readonly runs: {
    name: string;
    raw: Float64Array;
    m: Float64Array;
    v: Float64Array;
    loss: number;
    z: Float64Array;
    history: number[];
  }[];

  constructor(
    readonly target: Float64Array,
    readonly space: Space = DEFAULT_SPACE,
    steps = 300,
    lr = 0.03,
  ) {
    this.steps = steps;
    this.lr = lr;
    const starts: Params[] = [TYPICAL.N, TYPICAL.S, TYPICAL.V, TYPICAL.F, MCSHARRY];
    this.runs = starts.map((p, i) => {
      const raw = toRaw(p, target[0], space);
      const { loss, z } = lossAndGrad(raw, target, space);
      return {
        name: START_NAMES[i],
        raw,
        m: new Float64Array(16),
        v: new Float64Array(16),
        loss,
        z,
        history: [loss],
      };
    });
  }

  get done() {
    return this.step >= this.steps;
  }

  /** Advance every run by up to n Adam steps (betas 0.9/0.999, cosine decay). */
  advance(n: number) {
    const b1 = 0.9;
    const b2 = 0.999;
    for (let s = 0; s < n && !this.done; s++) {
      const lr = (this.lr * (1 + Math.cos((Math.PI * this.step) / this.steps))) / 2;
      const t = this.step + 1;
      for (const run of this.runs) {
        const { grad } = lossAndGrad(run.raw, this.target, this.space);
        for (let i = 0; i < 16; i++) {
          run.m[i] = b1 * run.m[i] + (1 - b1) * grad[i];
          run.v[i] = b2 * run.v[i] + (1 - b2) * grad[i] * grad[i];
          const mh = run.m[i] / (1 - b1 ** t);
          const vh = run.v[i] / (1 - b2 ** t);
          run.raw[i] -= (lr * mh) / (Math.sqrt(vh) + 1e-8);
        }
        const after = lossAndGrad(run.raw, this.target, this.space);
        run.loss = after.loss;
        run.z = after.z;
        run.history.push(after.loss);
      }
      this.step++;
    }
  }

  best() {
    let best = this.runs[0];
    for (const r of this.runs) if (r.loss < best.loss) best = r;
    return {
      name: best.name,
      params: rawToParams(best.raw, this.space),
      z0: best.raw[Z0],
      z: best.z,
      loss: best.loss,
      r2: r2(best.z, this.target),
    };
  }
}
