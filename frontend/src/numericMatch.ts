/**
 * Port of cli._loose_numeric_match — must match the CLI verdict exactly.
 */
export function looseNumericMatch(pred: string, gold: string): boolean {
  const clean = (s: string) =>
    s.trim().replaceAll('$', '').replaceAll(',', '').replaceAll('%', '').trim();
  const p = parseFloat(clean(pred));
  const g = parseFloat(clean(gold));
  if (Number.isFinite(p) && Number.isFinite(g)) return Math.round(p) === Math.round(g);
  return clean(pred).toLowerCase() === clean(gold).toLowerCase();
}
