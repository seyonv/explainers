import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";

test("study-build skips non-folder arguments (the paper-* glob also matches paper-template.html)", () => {
  const out = execFileSync("node", ["study-build.mjs", "paper-template.html"], { cwd: new URL("..", import.meta.url).pathname, encoding: "utf8" });
  assert.match(out, /study: 0 built, 0 stripped/);
});
