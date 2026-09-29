// prepack hook for the published packages: copies the repo-root LICENSE and NOTICE (Apache-2.0
// attribution for the vendored laya-ts) into the package directory so they ship in the tarball.
// Plain Node on purpose: it runs under `bun pm pack` and `npm pack` alike. The copies are
// gitignored; `scripts/pack-check.ts` fails if a tarball is missing them.
import { copyFileSync } from "node:fs";

for (const file of ["LICENSE", "NOTICE"]) {
  copyFileSync(new URL(`../${file}`, import.meta.url), file);
}
