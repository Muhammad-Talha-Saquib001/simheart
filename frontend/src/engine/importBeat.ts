import { N_SAMPLES, R_PEAK_INDEX } from "./ecgsyn";

export const MAX_BEAT_FILE_BYTES = 16_384;
/** The R peak may sit this many samples (about 28 ms) either side of row 73. */
const PEAK_WINDOW = 10;
/** Within that window the beat must reach this share of its largest deflection. */
const PEAK_SHARE = 0.4;
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

/**
 * A warning when the beat's main deflection is not near row 73 (sample 72), or null.
 *
 * Deflection = distance from the beat's median, so negative R waves count too. On the
 * bundled MIT-BIH validation beats, 100 % of S and V and 24 of 25 F beats pass, against
 * at most 1 % of the same beats shifted by 40 samples. Because a few real beats fail,
 * this is a warning: the fit still runs.
 */
export function alignmentWarning(beat: ArrayLike<number>): string | null {
  const sorted = Array.from(beat).sort((a, b) => a - b);
  const median = (sorted[(sorted.length - 1) >> 1] + sorted[sorted.length >> 1]) / 2;
  let near = 0;
  let largest = 0;
  for (let i = 0; i < beat.length; i++) {
    const d = Math.abs(beat[i] - median);
    largest = Math.max(largest, d);
    if (Math.abs(i - R_PEAK_INDEX) <= PEAK_WINDOW) near = Math.max(near, d);
  }
  if (near >= PEAK_SHARE * largest) return null;
  return "The largest deflection isn't near row 73. Check the beat is aligned on its R peak; the fit may be poor.";
}
