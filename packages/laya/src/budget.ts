// Per-call token budget for predict/systemOne/predictBatch (plan, Phase 6).

/** Ceiling on a per-call `maxLen` / `headMaxLen` (Python server `LAYA_MAX_TOKEN_BUDGET` default). */
export const MAX_TOKEN_BUDGET = 8192;

/**
 * Validate a per-call token budget the way Python's server validates `max_len` / `head_max_len`
 * (`serve.py` `_validate_budget_param`): null/undefined means unset, otherwise a positive
 * integer no larger than `MAX_TOKEN_BUDGET`.
 */
export function checkTokenBudget(name: string, value: unknown): number | null {
  if (value === null || value === undefined) return null;
  if (typeof value !== "number" || !Number.isInteger(value)) throw new TypeError(`${name} must be an integer`);
  if (value <= 0) throw new RangeError(`${name} must be a positive integer`);
  if (value > MAX_TOKEN_BUDGET) {
    throw new RangeError(`${name} exceeds the token budget (${value} > ${MAX_TOKEN_BUDGET})`);
  }
  return value;
}
