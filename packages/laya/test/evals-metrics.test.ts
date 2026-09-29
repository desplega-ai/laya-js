// The Phase 4 TS metrics match the Python ones (upstream `laya.evals` + bench_local.py) to 1e-9
// on a fixed 20-row fixture written by evals/python/make_metrics_fixture.py.
import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { type EvalRecord, ece, type Metrics, type Sliced, sliced } from "../evals/metrics.js";

const fixture = JSON.parse(
  readFileSync(new URL("../../../evals/fixtures/metrics-20.json", import.meta.url), "utf8"),
) as {
  records: EvalRecord[];
  expected: Sliced;
};

function expectClose(got: Metrics, want: Metrics) {
  expect(Object.keys(got).sort()).toEqual(Object.keys(want).sort());
  for (const k of Object.keys(want)) expect(Math.abs(got[k] - want[k]), k).toBeLessThanOrEqual(1e-9);
}

describe("evals metrics", () => {
  const got = sliced(fixture.records);

  it("matches Python overall", () => expectClose(got.overall, fixture.expected.overall));

  for (const key of ["by_suite", "by_tag", "by_language"] as const) {
    it(`matches Python ${key}`, () => {
      expect(Object.keys(got[key])).toEqual(Object.keys(fixture.expected[key]));
      for (const s of Object.keys(fixture.expected[key])) expectClose(got[key][s], fixture.expected[key][s]);
    });
  }

  it("puts a confidence on a bin edge in the lower bin, and 0 in the first", () => {
    // 0.2 closes bin 2 ((2/15, 3/15]) with a hit, 0 opens bin 0 with a miss: 0.5·0.8 + 0.5·0.
    expect(ece([0.2, 0], [true, false])).toBeCloseTo(0.5 * 0.8 + 0.5 * 0, 12);
  });
});
