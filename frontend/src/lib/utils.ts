import { type ClassValue, clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Merge class names, letting a later utility win over an earlier one of the
 * same kind (`cn('p-2', 'p-4')` -> `'p-4'`). Every vendored shadcn component
 * routes its `className` prop through this, which is what makes them
 * overridable at the call site instead of fighting specificity.
 */
export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}
