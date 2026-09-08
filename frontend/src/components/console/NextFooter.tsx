import { ArrowLeft, ArrowRight } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { nextItem, prevItem } from '../../app/nav';

/**
 * Every page ends in a next step.
 *
 * The story order lives in `nav.ts`; this reads the page before and after the
 * current one and offers both, with the hint that says why a reader would go
 * there. The last page in the order gets only a "back", never a dead end.
 */
export function NextFooter() {
  const { pathname } = useLocation();
  const next = nextItem(pathname);
  const prev = prevItem(pathname);
  if (!next && !prev) return null;

  return (
    <nav
      aria-label="Next and previous page"
      data-testid="next-footer"
      className="mt-6 grid gap-2 border-t border-line pt-4 sm:grid-cols-2"
    >
      {prev ? (
        <Link
          to={prev.to}
          className="group flex min-w-0 items-start gap-2.5 rounded-md border border-line bg-panel p-3 transition-colors hover:border-amber-line hover:bg-panel-2"
        >
          <ArrowLeft className="mt-1 size-4 shrink-0 text-faint group-hover:text-amber" aria-hidden />
          <span className="min-w-0">
            <span className="mono-caps block">previous</span>
            <span className="type-body block font-medium text-text">{prev.label}</span>
            <span className="type-meta block">{prev.hint}</span>
          </span>
        </Link>
      ) : (
        <span />
      )}
      {next && (
        <Link
          to={next.to}
          data-testid="next-link"
          className="group flex min-w-0 items-start justify-end gap-2.5 rounded-md border border-line bg-panel p-3 text-right transition-colors hover:border-amber-line hover:bg-panel-2"
        >
          <span className="min-w-0">
            <span className="mono-caps block">next</span>
            <span className="type-body block font-medium text-text">{next.label}</span>
            <span className="type-meta block">{next.hint}</span>
          </span>
          <ArrowRight className="mt-1 size-4 shrink-0 text-faint group-hover:text-amber" aria-hidden />
        </Link>
      )}
    </nav>
  );
}
