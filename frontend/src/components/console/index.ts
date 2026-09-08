/**
 * The Console's shared components — the one module every feature imports its
 * chrome from. `features/admin/ui.tsx` re-exports these for the admin pages
 * that predate this module; new code imports from here.
 */
export { PageHeader } from './PageHeader';
export { NextFooter } from './NextFooter';
export { Lamp, LampRow } from './Lamp';
export type { LampProps, LampTone } from './Lamp';
export { FilterChip } from './FilterChip';
