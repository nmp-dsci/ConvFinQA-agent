import type { ReactNode } from 'react';
import { ArrowUpRight } from 'lucide-react';
import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { Sparkline } from './Sparkline';
import type { SparkTone } from './Sparkline';
import { NO_VALUE } from './format';

export interface HudTileProps {
  /** The mono-caps micro-label above the number. */
  label: string;
  /** Already formatted. `NO_VALUE` (an em dash) when nothing was measured. */
  value: string;
  /** The line under the number when the value exists. */
  meta?: ReactNode;
  /**
   * Why the value is missing. Required whenever `value` is an em dash: a tile
   * that shows nothing and explains nothing is worse than no tile.
   */
  reason?: string;
  tone?: SparkTone | 'plain';
  series?: Array<number | null | undefined>;
  /**
   * The read has not come back yet. Distinct from an absent value: "we are
   * still asking" and "we asked and there is nothing" are different facts and
   * must not share a rendering.
   */
  loading?: boolean;
  /** The page where this number's rows live. Every tile has one. */
  to: string;
  /** How that page is named, printed on the tile so the link is legible. */
  drill: string;
}

const VALUE_TONE: Record<string, string> = {
  amber: 'text-text',
  good: 'text-good',
  bad: 'text-bad',
  info: 'text-info',
  plain: 'text-text',
};

/**
 * One number on the board.
 *
 * Every tile is a link, without exception. A metric a reader cannot drill into
 * is decoration — if a figure is worth this much visual weight it is worth
 * being able to see the rows behind it, and a tile with nowhere to go is a
 * tile that should not have been built.
 *
 * The absent state is a first-class rendering, not a fallback: `—` plus a
 * reason, no sparkline, muted. `/metrics/production` returns `null` with
 * `n_measured: 0` for latency, tokens and cost until someone runs a metered
 * eval, so this is the *normal* appearance of half this board today.
 */
export function HudTile({
  label,
  value,
  meta,
  reason,
  tone = 'plain',
  series,
  loading = false,
  to,
  drill,
}: HudTileProps) {
  const absent = !loading && value === NO_VALUE;
  const sparkTone: SparkTone = tone === 'plain' ? 'amber' : tone;

  return (
    <Link
      to={to}
      data-testid={`hud-tile-${label.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`}
      data-absent={absent ? 'true' : 'false'}
      className={cn(
        'group flex min-w-0 flex-col gap-1.5 rounded-md border border-line bg-panel p-3',
        'transition-colors hover:border-amber-line hover:bg-panel-2',
        'focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber',
      )}
    >
      <div className="flex items-start justify-between gap-2">
        <span className="mono-caps">{label}</span>
        <ArrowUpRight
          aria-hidden
          className="size-3 shrink-0 text-faint transition-colors group-hover:text-amber"
        />
      </div>

      {loading ? (
        <div className="h-[27px] w-20 animate-pulse rounded bg-panel-2" aria-label="loading" />
      ) : (
        <div
          data-testid="hud-value"
          className={cn('type-hud truncate', absent ? 'text-faint' : VALUE_TONE[tone])}
        >
          {value}
        </div>
      )}

      {loading ? (
        <div className="h-3 w-full animate-pulse rounded bg-panel-2" />
      ) : absent ? (
        <p data-testid="hud-reason" className="type-meta text-faint">
          {reason ?? 'not measured'}
        </p>
      ) : (
        meta && <div className="type-meta">{meta}</div>
      )}

      {series && (
        <div className="pt-0.5">
          {absent || loading ? (
            <Sparkline values={[]} tone={sparkTone} emptyLabel="no series" />
          ) : (
            <Sparkline values={series} tone={sparkTone} />
          )}
        </div>
      )}

      {/*
        A path, not a shout — so `.mono-caps` (which uppercases) is not used
        here. The tile names the page its rows live on, in the form the reader
        will see in the address bar.
      */}
      {/*
        Full `text-faint`, not `text-faint/80`: at 80% opacity over `--panel`
        this line measured 4.37:1 in dark and 3.72:1 in light, under the 4.5:1
        AA threshold, even after the token itself was corrected. Alpha on top of
        an already-quiet colour is how a contrast fix silently fails to apply.
      */}
      <span className="type-num mt-auto pt-1.5 text-[10px] tracking-[0.06em] text-faint group-hover:text-amber">
        {drill}
      </span>
    </Link>
  );
}
