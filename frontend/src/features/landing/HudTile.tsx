import type { ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Sparkline } from './Sparkline';
import type { SparkTone } from './Sparkline';
import { NO_VALUE } from './format';

export interface HudTileProps {
  /** The mono uppercase label above the number. */
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

/**
 * One number on the board, as a house stat tile that is also a link: a metric a reader cannot
 * drill into is decoration. The absent state is a first-class rendering — `—` plus the reason,
 * no sparkline — because `/metrics/production` returns null until a metered eval runs.
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
  const valueTone = absent ? 'absent' : tone === 'good' ? 'ok' : tone === 'bad' ? 'warn' : '';

  return (
    <Link
      to={to}
      data-testid={`hud-tile-${label.replace(/[^a-z0-9]+/gi, '-').toLowerCase()}`}
      data-absent={absent ? 'true' : 'false'}
      className="kpi"
    >
      <span className="label">{label}</span>
      {loading ? (
        <div className="n absent" aria-busy="true">
          …
        </div>
      ) : (
        <div data-testid="hud-value" className={`n ${valueTone}`}>
          {value}
        </div>
      )}
      {loading ? null : absent ? (
        <p data-testid="hud-reason" className="b">
          {reason ?? 'not measured'}
        </p>
      ) : (
        meta && <div className="b">{meta}</div>
      )}
      {series &&
        (absent || loading ? (
          <Sparkline values={[]} tone={sparkTone} emptyLabel="no series" />
        ) : (
          <Sparkline values={series} tone={sparkTone} />
        ))}
      <span className="drill">{drill} →</span>
    </Link>
  );
}
