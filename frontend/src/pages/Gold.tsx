import { useMemo } from 'react';
import { Link } from 'react-router-dom';
import type { ColumnDef } from '@tanstack/react-table';
import { fmtPct } from '@/kit/format';
import { Clip, Empty, Kpi, Loading, PageHead, Status } from '@/kit/ui';
import type { StatusTone } from '@/kit/ui';
import { encPath, useLens } from '@/kit/url';
import { InstrumentTable } from '@/features/admin/InstrumentTable';
import { useGoldReview } from '@/features/admin/goldApi';
import type { GoldReviewCase } from '@/features/admin/goldApi';
import { versionLabel } from '@/features/admin/lib';
import './gold.css';

/**
 * Slot 8 · what we know by hand.
 *
 * Attribution is gold-derived, so a wrong gold label bills an innocent agent. The loop never
 * lets that happen silently: the rule charges no agent when the gold's operands are absent from
 * the filing (`gold_suspect`) or when the record cannot say who failed (`ambiguous`, settled by
 * one binary question to the teacher — `teacher.resolve_ambiguous`), and the teacher can flag the
 * gold or dissent from the rule on any case. This page is every such case, read-only, straight
 * from the append-only diagnoses ledger: the rows a human settles by hand.
 */

const FILTERS = [
  { key: 'all', label: 'all' },
  { key: 'suspect', label: 'gold suspect' },
  { key: 'ambiguous', label: 'ambiguous' },
  { key: 'adjudicated', label: 'adjudicated' },
  { key: 'disputed', label: 'disputed' },
] as const;
type FilterKey = (typeof FILTERS)[number]['key'];

const isFilter = (s: string | null): s is FilterKey => FILTERS.some((f) => f.key === s);

const KEEP: Record<FilterKey, (c: GoldReviewCase) => boolean> = {
  all: () => true,
  suspect: (c) => c.gold_suspect || c.derived_agent === 'gold_suspect',
  ambiguous: (c) => c.derived_agent === 'ambiguous',
  adjudicated: (c) => c.adjudicated,
  disputed: (c) => c.attribution_disputed,
};

const chatPath = (reportId: string) => `/agent/chat/${encPath(reportId)}`;

/** One verdict per case, most settled first: a status word and its glyph, never colour alone. */
function verdictOf(c: GoldReviewCase): { tone: StatusTone; word: string; note: string } {
  if (c.adjudicated)
    return { tone: 'ok', word: `settled · ${c.derived_agent || '—'}`, note: 'the binary question named the agent' };
  if (c.derived_agent === 'gold_suspect')
    return { tone: 'err', word: 'gold looks wrong', note: "the rule found gold's operands absent from the filing" };
  if (c.derived_agent === 'ambiguous')
    return { tone: 'no', word: 'ambiguous · unsettled', note: 'no agent charged; the record cannot say who failed' };
  return {
    tone: 'warn',
    word: `teacher doubts gold`,
    note: `the rule still charged ${c.derived_agent || '—'}`,
  };
}

function Answer({ value, program, pred }: { value: string; program?: string; pred?: boolean }) {
  return (
    <>
      <span className={`ans${pred ? ' pred' : ''}${value ? '' : ' none'}`}>{value || '— none given'}</span>
      {program && <span className="prog">{program}</span>}
    </>
  );
}

export default function Gold() {
  const query = useGoldReview();
  const [lens, setLens] = useLens();
  const rawShow = lens.get('show');
  const show: FilterKey = isFilter(rawShow) ? rawShow : 'all';

  const cases = useMemo(() => query.data?.cases ?? [], [query.data]);
  const counts = useMemo(
    () => Object.fromEntries(FILTERS.map((f) => [f.key, cases.filter(KEEP[f.key]).length])) as Record<FilterKey, number>,
    [cases],
  );
  const rows = useMemo(() => cases.filter(KEEP[show]), [cases, show]);

  const columns = useMemo<Array<ColumnDef<GoldReviewCase, unknown>>>(
    () => [
      {
        header: 'question',
        accessorKey: 'question_id',
        meta: { align: 'left', mono: false, wrap: true, width: '17%' },
        cell: ({ row }) => {
          const c = row.original;
          return (
            <>
              <Link className="qid" to={chatPath(c.report_id)} title="ask the agent about this filing">
                {c.question_id || c.report_id}
              </Link>
              <span className="path">
                {versionLabel(c.version) || '—'} · {c.runtime || '—'} · {c.split || '—'}
              </span>
            </>
          );
        },
      },
      {
        header: 'asked',
        accessorKey: 'question',
        enableSorting: false,
        meta: { align: 'left', mono: false, wrap: true, width: '20%' },
        cell: ({ row }) => <Clip text={row.original.question} at={120} />,
      },
      {
        header: 'gold',
        accessorKey: 'gold_answer',
        enableSorting: false,
        meta: { align: 'right', wrap: true, width: '11%' },
        cell: ({ row }) => <Answer value={row.original.gold_answer} program={row.original.gold_program} />,
      },
      {
        header: 'predicted',
        accessorKey: 'pred_answer',
        enableSorting: false,
        meta: { align: 'right', wrap: true, width: '10%' },
        cell: ({ row }) => <Answer value={row.original.pred_answer} program={row.original.pred_program} pred />,
      },
      {
        header: 'verdict',
        id: 'verdict',
        accessorFn: (c) => verdictOf(c).word,
        meta: { align: 'left', mono: false, wrap: true, width: '15%' },
        cell: ({ row }) => {
          const c = row.original;
          const v = verdictOf(c);
          return (
            <span className="verdict">
              <Status tone={v.tone} title={v.note}>
                {v.word}
              </Status>
              <span className="tags">
                {c.gold_suspect && <span className="tag warn">flagged</span>}
                {c.attribution_disputed && <span className="tag">disputed</span>}
                {c.stage && <span className="tag">teacher: {c.stage}</span>}
              </span>
            </span>
          );
        },
      },
      {
        header: 'why',
        id: 'why',
        enableSorting: false,
        meta: { align: 'left', mono: false, wrap: true, width: '27%' },
        cell: ({ row }) => {
          const c = row.original;
          const reasons: Array<[string, string]> = [];
          if (c.adjudication_reason) reasons.push(['adjudication', c.adjudication_reason]);
          if (c.attribution_reason) reasons.push(['teacher on the attribution', c.attribution_reason]);
          if (!reasons.length && c.what_went_wrong) reasons.push(['what went wrong', c.what_went_wrong]);
          return (
            <>
              {reasons.length ? (
                reasons.map(([k, text]) => (
                  <div className="why" key={k}>
                    <span className="k">{k}</span>
                    <Clip text={text} at={180} />
                  </div>
                ))
              ) : (
                <span className="muted">— no reason recorded on this diagnosis</span>
              )}
              {c.missing_gold_operands.length > 0 && (
                <div className="miss">gold operands not found: {c.missing_gold_operands.join(', ')}</div>
              )}
            </>
          );
        },
      },
    ],
    [],
  );

  if (!query.data) {
    return (
      <div data-testid="gold-route">
        <PageHead
          eyebrow="gold review · what we know by hand"
          title={
            <>
              Gold labels the loop could not <em>trust</em>, and how each was settled
            </>
          }
        />
        <Loading what="/eval/gold-review" error={query.error ? (query.error as Error).message : null} />
      </div>
    );
  }

  const { summary: s, reason, source } = query.data;
  const settledOf = s.n_adjudicated + s.n_ambiguous;

  return (
    <div data-testid="gold-route">
      <PageHead
        eyebrow="gold review · what we know by hand"
        title={
          s.n_diagnoses ? (
            <>
              {s.n_gold_suspect_flags} of the {s.n_diagnoses} diagnoses doubted the gold, and {s.n_adjudicated} cases
              needed a <em>binary</em> adjudication
            </>
          ) : (
            <>
              No diagnosis has <em>doubted</em> the gold yet
            </>
          )
        }
        lead={
          <>
            Attribution is read from gold, so a wrong gold label would bill an innocent agent. These are the{' '}
            {s.n_review_rows} diagnoses, on {s.n_review_questions} questions, where the teacher flagged the gold, the
            rule charged no agent, or one yes/no question had to settle who failed. Read-only: settle a case by fixing
            the dataset, not here.
          </>
        }
      />

      {reason ? (
        <Empty testId="gold-empty">{reason}</Empty>
      ) : (
        <>
          <div className="kpis">
            <Kpi
              n={`${s.n_gold_suspect_flags}/${s.n_diagnoses}`}
              b={<>diagnoses where the teacher said the gold looks wrong ({fmtPct(s.n_gold_suspect_flags / s.n_diagnoses)})</>}
              tone="warn"
            />
            <Kpi
              n={`${s.n_derived_gold_suspect}/${s.n_diagnoses}`}
              b={<>where the rule itself found gold's operands absent from the filing, and charged no agent</>}
            />
            <Kpi
              n={`${s.n_adjudicated}/${settledOf}`}
              b={<>ambiguous cases settled by the binary question; {s.n_ambiguous} still charge no agent</>}
              tone="ok"
            />
            <Kpi
              n={`${s.n_disputed}/${s.n_diagnoses}`}
              b={<>diagnoses where the teacher dissented from the gold-derived attribution</>}
            />
          </div>

          <section>
            <h2>Every doubted case, with the gold beside what the agent said</h2>
            <div className="filters" role="group" aria-label="which cases">
              {FILTERS.map((f) => (
                <button
                  key={f.key}
                  type="button"
                  className={`tog${f.key === show ? ' on' : ''}`}
                  aria-pressed={f.key === show}
                  data-testid={`gold-filter-${f.key}`}
                  onClick={() => setLens({ show: f.key === 'all' ? null : f.key })}
                >
                  {f.label} · {counts[f.key]}
                </button>
              ))}
              <span className="count">
                {rows.length} of {cases.length} rows
              </span>
            </div>
            {rows.length === 0 ? (
              <Empty>no {FILTERS.find((f) => f.key === show)?.label} cases in the ledger</Empty>
            ) : (
              <div className="gold-table">
                <InstrumentTable
                  data={rows}
                  columns={columns}
                  rowKey={(c) => c.diagnosis_id}
                  minWidth={960}
                  layout="fixed"
                  maxHeight={760}
                  testId="gold-table"
                />
              </div>
            )}
            <p className="gold-source">
              Source: <code>/eval/gold-review</code>, read from the append-only ledger <code>{source}</code>. A question
              appears once per prompt version that failed it; no model is called to build this page.
            </p>
          </section>
        </>
      )}
    </div>
  );
}
