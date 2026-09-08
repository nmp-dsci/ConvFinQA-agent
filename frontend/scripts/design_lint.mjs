#!/usr/bin/env node
/**
 * The design lint — the never-do list from DESIGN.md as a check.
 *
 * Two kinds of rule:
 *
 *  1. Hard failures: a hex colour outside `tokens.css` and a legacy alias
 *     (`text-accent`, `bg-bg`, `text-textMuted`…). These have no baseline; one
 *     occurrence fails the build. An inline `style` is allowed as long as its
 *     value is a token — the hex rule catches a literal either way.
 *
 *  2. Ratchets: raw `text-[Npx]` sizes, and the subset under 11px. Every file's
 *     count is held to `design-baseline.json` and may only go down. Run with
 *     `--update` after removing sizes so the baseline follows. A file that is
 *     not in the baseline is held to zero, so a new component cannot start with
 *     raw sizes.
 *
 * Runs in CI from the frontend job. No browser, no dependencies — a probe that
 * renders pages and measures computed styles is a separate script.
 */
import { readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const srcDir = join(root, 'src');
const baselinePath = join(root, 'design-baseline.json');
const update = process.argv.includes('--update');

function walk(dir, out = []) {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) walk(p, out);
    else if (/\.(tsx?|css)$/.test(name) && !/\.test\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

// `(?<!&)` skips HTML entities such as `&#123;` in JSX text.
const HEX = /(?<!&)#[0-9a-fA-F]{6}\b|(?<!&)#[0-9a-fA-F]{3}\b(?![0-9a-fA-F])/g;
const LEGACY =
  /\b(text-accent|border-accent|bg-bg|text-textMuted|text-textMain|bg-panel2|text-accent2|text-danger|bg-bubbleUser|bg-bubbleAssistant)\b/g;
const RAW_PX = /\btext-\[(\d+(?:\.\d+)?)px\]/g;

const hard = [];
const counts = {};
const small = {};
let totalRaw = 0;
let totalSmall = 0;

for (const file of walk(srcDir)) {
  const rel = relative(root, file);
  const text = readFileSync(file, 'utf8');
  const isTokens = rel.endsWith('tokens.css');
  const isVendored = rel.startsWith('src/components/ui/');

  if (!isTokens) {
    for (const m of text.matchAll(HEX)) hard.push(`${rel}: hex colour ${m[0]} — use a token`);
  }
  // The vendored shadcn files carry `bg-accent`, which resolves to a token via
  // the alias block; they are excluded from the alias rule but not the others.
  if (!isVendored) {
    for (const m of text.matchAll(LEGACY)) hard.push(`${rel}: legacy alias ${m[1]}`);
  }

  let n = 0;
  let s = 0;
  for (const m of text.matchAll(RAW_PX)) {
    n += 1;
    if (Number.parseFloat(m[1]) < 11) s += 1;
  }
  if (n) counts[rel] = n;
  if (s) small[rel] = s;
  totalRaw += n;
  totalSmall += s;
}

let baseline = { raw_px: {}, under_11px: {} };
try {
  baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
} catch {
  if (!update) hard.push('design-baseline.json is missing — run `npm run lint:design -- --update`');
}

const ratchet = [];
if (!update) {
  for (const [file, n] of Object.entries(counts)) {
    const allowed = baseline.raw_px[file] ?? 0;
    if (n > allowed) ratchet.push(`${file}: ${n} raw text-[Npx] sizes, baseline ${allowed}`);
  }
  for (const [file, n] of Object.entries(small)) {
    const allowed = baseline.under_11px[file] ?? 0;
    if (n > allowed) ratchet.push(`${file}: ${n} sizes under 11px, baseline ${allowed}`);
  }
}

if (update) {
  writeFileSync(baselinePath, `${JSON.stringify({ raw_px: counts, under_11px: small }, null, 2)}\n`);
  console.log(`design-baseline.json updated: ${totalRaw} raw sizes, ${totalSmall} under 11px`);
}

console.log(`design lint: ${totalRaw} raw text-[Npx] sizes (${totalSmall} under 11px) across ${Object.keys(counts).length} files`);
for (const line of hard) console.log(`  ✗ ${line}`);
for (const line of ratchet) console.log(`  ✗ ${line}`);

if (hard.length || ratchet.length) {
  console.log(`\ndesign lint failed: ${hard.length} hard, ${ratchet.length} ratchet`);
  process.exit(1);
}
console.log('design lint: clean');
