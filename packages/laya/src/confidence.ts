// Port of `laya/confidence.py` (#361): opt-in abstention for predict/systemOne/predictBatch.

/** Validate an abstention threshold: a finite number in [0, 1]. Booleans and strings are rejected. */
export function checkMinConfidence(v: unknown): number {
  if (typeof v !== "number" || !Number.isFinite(v) || v < 0 || v > 1) {
    throw new RangeError(`minConfidence must be a number in [0.0, 1.0], got ${reprOf(v)}`);
  }
  return v;
}

/**
 * Add `low_confidence: true` to every answer whose `answer_confidence` (falling back to
 * `confidence`) is below `minConfidence`. The raw answer is left intact. `0` is a no-op.
 */
export function flagLowConfidence(results: unknown[], minConfidence: number): void {
  if (minConfidence === 0) return;
  for (const res of results) {
    const answers = isPlainObject(res) ? res.answers : undefined;
    if (!isPlainObject(answers)) continue;
    for (const a of Object.values(answers)) {
      if (!isPlainObject(a)) continue;
      const conf = a.answer_confidence ?? a.confidence;
      if (typeof conf === "number" && Number.isFinite(conf) && conf < minConfidence) a.low_confidence = true;
    }
  }
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function reprOf(v: unknown): string {
  if (typeof v === "number") return String(v);
  if (v === undefined) return "undefined";
  return JSON.stringify(v) ?? String(v);
}
