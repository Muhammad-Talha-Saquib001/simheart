import { N_SAMPLES } from "./ecgsyn";

export const MAX_BEAT_FILE_BYTES = 16_384;
const SIM_MIN = -0.01563;
const SIM_MAX = 0.042557;
const DECIMAL = /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i;

/** Read a single, already aligned ECG beat in mV and put it on the simulator scale. */
export function importBeat(csv: string): Float64Array {
  const rows = csv.trim().split(/\r?\n/).map((row) => row.trim());
  if (rows[0].toLowerCase() === "amplitude_mv") rows.shift();
  if (rows.length !== N_SAMPLES) {
    throw new Error(`Expected ${N_SAMPLES} amplitude rows. Found ${rows.length}.`);
  }
  const samples = rows.map((row, index) => {
    const value = Number(row);
    if (!DECIMAL.test(row) || !Number.isFinite(value)) {
      throw new Error(`Sample ${index + 1} must contain one finite number in mV.`);
    }
    return value;
  });
  const low = Math.min(...samples);
  const span = Math.max(...samples) - low;
  if (!Number.isFinite(span) || span < 1e-8) {
    throw new Error("The beat must have a finite, non-flat amplitude range.");
  }
  return Float64Array.from(samples, (value) => SIM_MIN + ((value - low) / span) * (SIM_MAX - SIM_MIN));
}
