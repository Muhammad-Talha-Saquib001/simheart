import { describe, expect, it } from "vitest";
import reference from "./__fixtures__/ecgsyn_reference.json";
import { MCSHARRY, N_SAMPLES, TYPICAL, decompose, simulate, type Params } from "./ecgsyn";
import { Calibrator, DEFAULT_SPACE, lossAndGrad, rawToParams, toRaw } from "./fitting";

const named: Record<string, Params> = { ...TYPICAL, McSharry: MCSHARRY };

describe("ecgsyn port", () => {
  it("matches the PyTorch simulator sample for sample", () => {
    for (const [name, ref] of Object.entries(reference.beats)) {
      const z = simulate(named[name], reference.z0);
      const worst = Math.max(...ref.map((v, i) => Math.abs(v - z[i])));
      expect(worst, name).toBeLessThan(1e-6);
    }
  });

  it("splits a beat into wave contributions that add back up", () => {
    const full = simulate(TYPICAL.N, 0);
    const base = simulate({ ...TYPICAL.N, P: { ...TYPICAL.N.P, a: 0 }, Q: { ...TYPICAL.N.Q, a: 0 }, R: { ...TYPICAL.N.R, a: 0 }, S: { ...TYPICAL.N.S, a: 0 }, T: { ...TYPICAL.N.T, a: 0 } }, 0);
    const parts = decompose(TYPICAL.N);
    for (let i = 0; i < N_SAMPLES; i++) {
      const sum = parts.reduce((s, p) => s + p[i], base[i]);
      expect(sum).toBeCloseTo(full[i], 10);
    }
  });
});

describe("calibration", () => {
  it("round-trips parameters through the raw space", () => {
    const back = rawToParams(toRaw(TYPICAL.V, 0, DEFAULT_SPACE), DEFAULT_SPACE);
    for (const w of ["P", "Q", "R", "S", "T"] as const) {
      expect(back[w].a).toBeCloseTo(TYPICAL.V[w].a, 8);
      expect(back[w].b).toBeCloseTo(TYPICAL.V[w].b, 6);
      expect(back[w].theta_deg).toBeCloseTo(TYPICAL.V[w].theta_deg, 5);
    }
  });

  it("has an exact gradient (checked against finite differences)", () => {
    const target = simulate(MCSHARRY, 0.001);
    const raw = toRaw(TYPICAL.F, 0, DEFAULT_SPACE);
    const { grad } = lossAndGrad(raw, target, DEFAULT_SPACE);
    for (let i = 0; i < 16; i++) {
      const h = 1e-6;
      const up = Float64Array.from(raw);
      const dn = Float64Array.from(raw);
      up[i] += h;
      dn[i] -= h;
      const fd = (lossAndGrad(up, target, DEFAULT_SPACE).loss - lossAndGrad(dn, target, DEFAULT_SPACE).loss) / (2 * h);
      expect(Math.abs(fd - grad[i]), `raw[${i}]`).toBeLessThan(1e-6 + 1e-4 * Math.abs(fd));
    }
  });

  it("recovers a simulated beat closely", () => {
    const target = simulate(TYPICAL.V, 0.0);
    const cal = new Calibrator(target);
    cal.advance(cal.steps);
    expect(cal.best().r2).toBeGreaterThan(0.97);
  });
});
