// Bounded working memory of the ONNX graph runs: row lengths, run planning, trimmed head feeds.
import { describe, expect, it } from "vitest";
import { type Batch, DEFAULT_RUN_MB, feedHead, planRuns, usedLengths } from "../src/providers.js";

const MB = 2 ** 20;

describe("usedLengths", () => {
  it("counts each row up to its last attended token", () => {
    expect(
      usedLengths([
        [1, 1, 1, 0, 0],
        [1, 0, 0, 0, 0],
        [1, 1, 1, 1, 1],
      ]),
    ).toEqual([3, 1, 5]);
  });
  it("never reports an empty row", () => {
    expect(usedLengths([[0, 0, 0], []])).toEqual([1, 1]);
  });
});

describe("planRuns", () => {
  it("keeps every row once, in order", () => {
    const lens = [40, 500, 12, 300, 300, 7, 900, 64];
    const runs = planRuns(lens, 64 * MB);
    expect(runs.flat()).toEqual(lens.map((_, i) => i));
  });
  it("puts all rows in one run when they fit or the budget is unbounded", () => {
    expect(planRuns([10, 20, 30], 64 * MB)).toEqual([[0, 1, 2]]);
    expect(planRuns([900, 900, 900], Number.POSITIVE_INFINITY)).toEqual([[0, 1, 2]]);
  });
  it("splits when the padded working set would pass the budget", () => {
    // 4 rows of 512 tokens need about 4 * 512^2 * 400 B = 400 MB: two runs of two at 256 MB.
    const runs = planRuns([512, 512, 512, 512], DEFAULT_RUN_MB * MB);
    expect(runs).toEqual([
      [0, 1],
      [2, 3],
    ]);
  });
  it("sizes a run by its longest row, so a long row closes the run it would bloat", () => {
    const runs = planRuns([16, 16, 16, 700, 16], 64 * MB);
    expect(runs).toEqual([[0, 1, 2], [3], [4]]);
  });
  it("runs a row larger than the budget on its own", () => {
    expect(planRuns([2000], MB)).toEqual([[0]]);
    expect(planRuns([2000, 2000], MB)).toEqual([[0], [1]]);
  });
  it("returns no runs for no rows", () => {
    expect(planRuns([], MB)).toEqual([]);
  });
});

describe("feedHead with a trimmed sequence length", () => {
  const ort = {
    Tensor: class {
      constructor(
        public type: string,
        public data: ArrayLike<unknown>,
        public dims: number[],
      ) {}
    },
  };
  const batch: Batch = {
    inputIds: [],
    attentionMask: [
      [1, 1, 1, 0, 0, 0],
      [1, 1, 0, 0, 0, 0],
    ],
    markerPos: [
      [0, 2],
      [1, 0],
    ],
    markerMask: [
      [true, true],
      [true, false],
    ],
    qtype: [0, 1],
  };
  // Row 0 keeps 3 tokens, row 1 keeps 2; the graph input is [2, 3, 2].
  const hidden = [
    [
      [1, 2],
      [3, 4],
      [5, 6],
    ],
    [
      [7, 8],
      [9, 10],
    ],
  ];
  it("zero-pads short rows to seqLen and trims the mask to match", () => {
    const out = feedHead(ort, hidden, batch, 3);
    expect(out.hidden_states.dims).toEqual([2, 3, 2]);
    expect(Array.from(out.hidden_states.data)).toEqual([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 0, 0]);
    expect(out.attention_mask.dims).toEqual([2, 3]);
    expect(Array.from(out.attention_mask.data)).toEqual([1n, 1n, 1n, 1n, 1n, 0n]);
  });
  it("still takes the width from the first row when seqLen is omitted", () => {
    const out = feedHead(ort, [hidden[0], hidden[0]], batch);
    expect(out.hidden_states.dims).toEqual([2, 3, 2]);
  });
});
