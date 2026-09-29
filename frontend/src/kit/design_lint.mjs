#!/usr/bin/env node
// nmp-ui-kit 0.1.0 · vendored from nmp-central-ai/ui/kit — do not edit here; change the kit and run `make ui-sync`.
/**
 * DESIGN.md's never-do list as a check. Walks src/ (this file lives in src/kit/):
 *   - a hex colour outside the two token files (kit/tokens.base.css, theme.css) fails;
 *   - a font-size under 12.8px (--t-2) fails, in CSS or in an inline style;
 *   - Inter, Roboto or Helvetica fail;
 *   - a Tailwind import or utility class list fails — the house is plain CSS on tokens.
 *     `--allow-tailwind` suspends that one rule while a project migrates onto the kit.
 * No dependencies; runs in CI.
 */
import { readdirSync, readFileSync, statSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const kitDir = dirname(fileURLToPath(import.meta.url));
const srcDir = join(kitDir, '..');
const root = join(srcDir, '..');
const allowTailwind = process.argv.includes('--allow-tailwind');
const TOKEN_FILES = new Set(['src/kit/tokens.base.css', 'src/theme.css']);

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.(tsx?|css)$/.test(name) && !/\.test\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

const HEX = /(?<![&\w])#[0-9a-fA-F]{6}\b|(?<![&\w])#[0-9a-fA-F]{3}\b(?![0-9a-fA-F])/g;
const SMALL_PX = /font-size:\s*(\d+(?:\.\d+)?)px/g;
const SMALL_INLINE = /fontSize:\s*['"]?(\d+(?:\.\d+)?)(?:px)?['"]?/g;
const BANNED_FONTS = /font-family:[^;]*\b(Inter|Roboto|Helvetica)\b/g;
const TAILWIND = /@import\s+['"]tailwindcss['"]|@tailwind\b|\b(?:text|bg|border)-\[[^\]]+\]/g;

const problems = [];
for (const file of walk(srcDir)) {
  const rel = relative(root, file).split('\\').join('/');
  const text = readFileSync(file, 'utf8');
  if (!TOKEN_FILES.has(rel)) {
    for (const m of text.matchAll(HEX)) problems.push(`${rel}: hex colour ${m[0]} — use a token`);
  }
  for (const m of text.matchAll(SMALL_PX)) {
    if (Number.parseFloat(m[1]) < 12.8) problems.push(`${rel}: font-size ${m[1]}px is under --t-2`);
  }
  for (const m of text.matchAll(SMALL_INLINE)) {
    if (Number.parseFloat(m[1]) < 12.8) problems.push(`${rel}: inline fontSize ${m[1]} is under --t-2`);
  }
  for (const m of text.matchAll(BANNED_FONTS)) problems.push(`${rel}: banned font ${m[1]}`);
  if (!allowTailwind) for (const m of text.matchAll(TAILWIND)) problems.push(`${rel}: Tailwind (${m[0]}) — the house is plain CSS on tokens`);
}
for (const p of problems) console.log(`  ✗ ${p}`);
if (problems.length) {
  console.log(`design lint failed: ${problems.length}`);
  process.exit(1);
}
console.log('design lint: clean');
