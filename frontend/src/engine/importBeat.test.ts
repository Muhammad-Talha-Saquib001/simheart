import { describe, expect, it } from "vitest";
import { N_SAMPLES, R_PEAK_INDEX, TYPICAL, cloneParams, simulate } from "./ecgsyn";
import { Calibrator, DEFAULT_SPACE } from "./fitting";
import { alignmentWarning, importBeat } from "./importBeat";

const example = Array.from({ length: N_SAMPLES }, (_, index) => index / 100);

describe("importBeat", () => {
  it("scales an aligned one-column beat and preserves sample order", () => {
    const csv = ["amplitude_mv", ...example].join("\r\n") + "\r\n";
    const beat = importBeat(csv);
    expect(beat).toHaveLength(N_SAMPLES);
    expect(beat[0]).toBeCloseTo(-0.01563);
    expect(beat[N_SAMPLES - 1]).toBeCloseTo(0.042557);
    expect(beat[R_PEAK_INDEX]).toBeGreaterThan(beat[R_PEAK_INDEX - 1]);
  });

  it("rejects incomplete, multi-column, invalid and flat recordings", () => {
    expect(() => importBeat(example.slice(1).join("\n"))).toThrow(/216 amplitude rows/);
    for (const invalid of ["0.1,0.2", "", "0x20", "1e999"]) {
      expect(() => importBeat([0, invalid, ...example.slice(2)].join("\n"))).toThrow(/Sample 2/);
    }
    expect(() => importBeat(example.map(() => 0.5).join("\n"))).toThrow(/non-flat/);
  });

  it("supports calibration and a new scenario from the imported samples", () => {
    const input = importBeat(Array.from(simulate(TYPICAL.V, 0), (value) => 2 * value + 0.1).join("\n"));
    const calibrator = new Calibrator(input, DEFAULT_SPACE, 300, 0.03);
    calibrator.advance(300);
    const fitted = calibrator.best();
    expect(fitted.r2).toBeGreaterThan(0.9);
    const changed = cloneParams(fitted.params);
    changed.T.a *= 2;
    const next = simulate(changed, fitted.z0);
    expect(next).toHaveLength(N_SAMPLES);
    expect(Math.max(...next.map((value, index) => Math.abs(value - fitted.z[index])))).toBeGreaterThan(0.001);
  });

  it("warns, without rejecting, when the main deflection is not at row 73", () => {
    const beat = Array.from(simulate(TYPICAL.V, 0));
    expect(alignmentWarning(importBeat(beat.join("\n")))).toBeNull();
    expect(alignmentWarning(importBeat(beat.map((value) => -value).join("\n")))).toBeNull();
    const shifted = beat.map((_, index) => beat[(index - 40 + N_SAMPLES) % N_SAMPLES]);
    expect(alignmentWarning(importBeat(shifted.join("\n")))).toMatch(/row 73/);
  });
});
