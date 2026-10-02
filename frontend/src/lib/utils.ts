/** Join class names, skipping falsy ones. The house is plain CSS, so there is nothing to merge. */
export function cn(...classes: Array<string | false | null | undefined>): string {
  return classes.filter(Boolean).join(' ');
}
