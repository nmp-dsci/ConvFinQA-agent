import { useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import type { ColumnDef } from '@tanstack/react-table';
import './optimise.css';
import { Status } from '@/kit/ui';
import type { StatusTone } from '@/kit/ui';
import { InstrumentTable } from './InstrumentTable';
import { clip, formatCount, formatStamp, relativeTime } from './lib';
import {
  AdminPage,
  Caveat,
  EmptyState,
  ErrorNote,
  Lamp,
  LampRow,
  LoadingRows,
  Panel,
  StatCells,
  Verdict,
  WriteGate,
} from './ui';
import { useResearchStatus, useRuleVariants, useRules } from './useAdminData';
import { cancelResearch, startResearch } from '../../lib/api';
import { qk } from '../../lib/queryClient';
import { useMode } from '../../modeStore';
import type { ResearchJob, Rule } from '../../types';

/**
 * Optimise › Rule research: the s7 prompt-improvement harness, from the outside.
 *
 * Two halves. The rules browser is the durable half: four append-only JSONL
 * stores, one per agent, holding the rules a diagnose → fix → verify loop
 * actually landed, plus every attempt it made — including the ones that failed
 * verification, which are the more interesting half and are not hidden.
 *
 * The console is the live half, and on the public demo it is inert: the launch
 * form is a real disabled fieldset and the server refuses `POST
 * /admin/research/start` regardless. It is shown rather than removed because a
 * visitor should be able to see what the operator can do, and that they cannot.
 */

const AGENTS = ['triage', 'preprocess', 'retriever', 'calculator'] as const;
type Agent = (typeof AGENTS)[number];

// ---------------------------------------------------------------------------
// Rule store helpers
// ---------------------------------------------------------------------------

function str(rule: Rule, key: string): string {
  const value = rule[key];
  if (value === null || value === undefined) return '';
  return typeof value === 'string' ? value : JSON.stringify(value);
}

function num(rule: Rule, key: string): number | null {
  const value = rule[key];
  return typeof value === 'number' ? value : null;
}

/** `[{report_id, turn_index}, …]` → `MAR/2010 · 1`, or the raw JSON. */
function caseRef(value: unknown): string {
  const one = Array.isArray(value) ? value[0] : value;
  if (one && typeof one === 'object' && 'report_id' in one) {
    const record = one as { report_id?: string; turn_index?: number };
    return `${record.report_id ?? ''} · ${record.turn_index ?? 0}`;
  }
  return '';
}

function jobTone(status: string): StatusTone {
  if (status === 'succeeded') return 'ok';
  if (status === 'failed') return 'err';
  if (status === 'running') return 'warn';
  return 'no';
}

// ---------------------------------------------------------------------------

function JobCard({ job }: { job: ResearchJob }) {
  return (
    <div className="card opt-job">
      <p className="label">
        {job.kind} · {job.job_id}
      </p>
      <h3>
        <Status tone={jobTone(job.status)}>{job.status}</Status>
      </h3>
      <p className="small muted">
        started {formatStamp(job.started_at)}
        {job.finished_at ? ` · finished ${formatStamp(job.finished_at)}` : ' · still running'}
        {job.returncode !== null ? ` · exit ${job.returncode}` : ''}
      </p>
      <p className="small mono wrap-any">
        {Object.entries(job.args)
          .map(([key, value]) => `${key}=${String(value)}`)
          .join(' · ')}
      </p>
      {job.log_tail.length > 0 && (
        <div className="code band">
          <p className="label">log tail · {job.log_tail.length} lines</p>
          <pre>{job.log_tail.join('\n')}</pre>
        </div>
      )}
    </div>
  );
}

function Section({ testId, n, label, claim, children }: { testId: string; n: number; label: string; claim: ReactNode; children: ReactNode }) {
  return (
    <section className="opt-sec" data-testid={testId}>
      <h2>
        {n} · {label} — {claim}
      </h2>
      {children}
    </section>
  );
}

// ---------------------------------------------------------------------------

export default function Research() {
  const isDemo = useMode((s) => s.health?.mode) === 'demo';
  const variants = useRuleVariants();
  const status = useResearchStatus();
  const queryClient = useQueryClient();

  // Empty means "whatever variant this backend is configured with" —
  // `/admin/rules` falls back to `settings.variant`. Defaulting to the last
  // entry in `/admin/rules/variants` instead would open on v3_2, whose stores
  // are legitimately empty, and read as a broken page.
  const [variant, setVariant] = useState('');
  const [agent, setAgent] = useState<Agent>('preprocess');
  const [view, setView] = useState<'rules' | 'attempts'>('rules');
  const [limit, setLimit] = useState(5);
  const [retryN, setRetryN] = useState(1);
  const [kind, setKind] = useState('s7');
  const [writeError, setWriteError] = useState<string | null>(null);

  const rules = useRules(variant);
  // The server tells us which variant it actually served, so the select can
  // show the truth on first paint without guessing at it.
  const activeVariant = rules.data?.variant ?? variant;
  const block = rules.data?.agents?.[agent];
  const rows = (view === 'rules' ? block?.rules : block?.attempts) ?? [];

  const canLaunch = status.data?.can_launch ?? false;
  const launchReason = isDemo
    ? 'Read-only demo. A research round runs the s7 harness against the live model — the server refuses this with a 501 (not_available_demo), and this container holds no API key to run it with.'
    : 'No OWNER_TOKEN is configured on this backend, so the launch route is refused with a 403 (owner_token_unset).';

  const start = useMutation({
    mutationFn: () => startResearch({ kind, limit, retry_n: retryN, variant: activeVariant }),
    onSuccess: () => {
      setWriteError(null);
      void queryClient.invalidateQueries({ queryKey: qk.researchStatus });
    },
    onError: (err) => setWriteError(err instanceof Error ? err.message : String(err)),
  });

  const cancel = useMutation({
    mutationFn: () => cancelResearch(),
    onSuccess: () => void queryClient.invalidateQueries({ queryKey: qk.researchStatus }),
    onError: (err) => setWriteError(err instanceof Error ? err.message : String(err)),
  });

  const columns = useMemo<Array<ColumnDef<Rule, unknown>>>(() => {
    const base: Array<ColumnDef<Rule, unknown>> = [
      {
        id: 'rule',
        header: 'rule',
        accessorFn: (r) => str(r, 'rule'),
        meta: { align: 'left', mono: false, wrap: true, width: '340px' },
        cell: ({ row }) => <span title={str(row.original, 'rule')}>{clip(str(row.original, 'rule'), 260)}</span>,
      },
      {
        id: 'fix_type',
        header: 'fix',
        accessorFn: (r) => str(r, 'fix_type'),
        meta: { align: 'left', width: '76px' },
      },
      {
        id: 'confidence',
        header: 'conf',
        accessorFn: (r) => num(r, 'confidence') ?? -1,
        cell: ({ row }) => {
          const value = num(row.original, 'confidence');
          return value === null ? <span className="muted">—</span> : value.toFixed(2);
        },
      },
    ];

    if (view === 'rules') {
      base.push(
        {
          id: 'verified_on',
          header: 'verified on',
          accessorFn: (r) => caseRef(r.verified_on),
          meta: { align: 'left', width: '190px' },
          cell: ({ row }) => <span className="muted">{caseRef(row.original.verified_on) || '—'}</span>,
        },
        {
          id: 'verified_at',
          header: 'verified',
          accessorFn: (r) => str(r, 'verified_at'),
          cell: ({ row }) => formatStamp(str(row.original, 'verified_at')),
        },
      );
    } else {
      base.push(
        {
          id: 'verify_result',
          header: 'result',
          accessorFn: (r) => str(r, 'verify_result'),
          meta: { align: 'left', width: '80px' },
          cell: ({ row }) => {
            const result = str(row.original, 'verify_result');
            return <Verdict ok={result === 'passed'}>{result || 'unknown'}</Verdict>;
          },
        },
        {
          id: 'attempted_on',
          header: 'attempted on',
          accessorFn: (r) => caseRef(r.attempted_on),
          meta: { align: 'left', width: '190px' },
          cell: ({ row }) => <span className="muted">{caseRef(row.original.attempted_on) || '—'}</span>,
        },
        {
          id: 'attempted_at',
          header: 'attempted',
          accessorFn: (r) => str(r, 'attempted_at'),
          cell: ({ row }) => formatStamp(str(row.original, 'attempted_at')),
        },
      );
    }
    return base;
  }, [view]);

  const counts = AGENTS.map((name) => ({
    name,
    rules: rules.data?.agents?.[name]?.rules.length ?? 0,
    attempts: rules.data?.agents?.[name]?.attempts.length ?? 0,
  }));

  const totalRules = counts.reduce((sum, c) => sum + c.rules, 0);
  const totalAttempts = counts.reduce((sum, c) => sum + c.attempts, 0);
  const agentCount = counts.find((c) => c.name === agent) ?? { rules: 0, attempts: 0 };

  const title: ReactNode = rules.isLoading ? (
    <>
      Reading the <em>rule</em> stores…
    </>
  ) : (
    <>
      {activeVariant || 'This variant'}&rsquo;s stores hold <em>{formatCount(totalRules)} landed rules</em> from{' '}
      {formatCount(totalAttempts)} attempts
    </>
  );

  const history = status.data?.history ?? [];

  return (
    <AdminPage
      testId="admin-research"
      eyebrow={`optimise · rule research · s7 harness · variant ${activeVariant || '…'}`}
      title={title}
      sub="The s7 harness: a per-case diagnose → route + fix → verify loop over the first wrong turn of each failing conversation. The rules it landed are below, with every attempt that failed verification; launching a new round is an operator action."
    >
      <LampRow>
        <Lamp
          label="round"
          value={status.data?.busy ? 'running' : 'idle'}
          tone={status.data?.busy ? 'good' : 'idle'}
          dashed={!status.data?.busy}
        />
        <Lamp
          label="launch"
          value={canLaunch ? 'enabled' : 'refused'}
          tone={canLaunch ? 'good' : 'idle'}
          dashed={!canLaunch}
          title={canLaunch ? undefined : launchReason}
        />
        <Lamp label="variant" value={activeVariant || '…'} tone="info" />
        <Lamp label="rules" value={formatCount(totalRules)} tone="idle" dashed />
        <Lamp label="attempts" value={formatCount(totalAttempts)} tone="idle" dashed />
      </LampRow>

      <Section
        testId="research-rules"
        n={1}
        label="Rules"
        claim={`${agent} landed ${formatCount(agentCount.rules)} of ${formatCount(agentCount.attempts)} attempts in ${activeVariant || '…'}`}
      >
        <p>
          Four append-only JSONL stores under <code>evaluation/diagnostics/</code>. The generated prompt
          module <code>src/convfinqa/prompts/{activeVariant || '…'}.py</code> is assembled from these — it
          is never hand-edited.
        </p>
        <div className="filters">
          <label className="pick">
            <span className="label">variant</span>
            <select value={activeVariant} onChange={(e) => setVariant(e.target.value)}>
              {(variants.data ?? []).map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </label>
          <div className="seg" role="group" aria-label="rules or attempts">
            {(['rules', 'attempts'] as const).map((key) => (
              <button
                key={key}
                type="button"
                aria-pressed={view === key}
                className={view === key ? 'on' : ''}
                onClick={() => setView(key)}
              >
                {key === 'rules' ? 'landed rules' : 'every attempt'}
              </button>
            ))}
          </div>
        </div>

        <div className="chips" role="group" aria-label="agent">
          {counts.map((c) => (
            <button
              key={c.name}
              type="button"
              aria-pressed={agent === c.name}
              onClick={() => setAgent(c.name)}
              className={`chip${agent === c.name ? ' on' : ''}`}
            >
              {c.name} · {view === 'rules' ? c.rules : c.attempts}
            </button>
          ))}
        </div>

        <StatCells
          columns={4}
          cells={counts.map((c) => ({
            label: `${c.name} · ${view === 'rules' ? 'landed' : 'attempts'}`,
            value: view === 'rules' ? `${c.rules} of ${c.attempts}` : `${c.attempts}`,
          }))}
        />

        {rules.isLoading ? (
          <LoadingRows rows={5} />
        ) : rules.error ? (
          <ErrorNote error={rules.error} />
        ) : (
          <InstrumentTable
            key={`${activeVariant}:${agent}:${view}`}
            data={rows}
            columns={columns}
            rowKey={(r, i) => str(r, 'rule_id') || str(r, 'attempt_id') || String(i)}
            minWidth={760}
            maxHeight={480}
            caption={`${view === 'rules' ? 'Landed rules' : 'Every attempt, failed verifications included'} for ${agent} in ${activeVariant || '…'}. Source: /admin/rules.`}
            emptyLabel={
              totalRules === 0 && totalAttempts === 0
                ? `${activeVariant} has empty rule stores: the round diagnosed its cases but landed no rule that survived verification, so nothing was written. That is a recorded outcome, not a missing file.`
                : view === 'rules'
                  ? `no rule landed for ${agent} in ${activeVariant}`
                  : `no attempt recorded for ${agent} in ${activeVariant}`
            }
          />
        )}

        <Caveat>
          Attempts include the ones that failed verification. A harness that only showed what worked
          would be a claim about its hit rate rather than a record of it — for{' '}
          {activeVariant || 'this variant'} the stores hold {formatCount(totalRules)} landed rules against{' '}
          {formatCount(totalAttempts)} attempts.
        </Caveat>
      </Section>

      <Panel
        testId="research-console"
        title={`Research console — ${canLaunch ? 'this backend can launch a round' : 'launching is refused here'}`}
        endpoint="POST /admin/research/start"
        note="a round runs the s7 harness as a subprocess against the live model — minutes, and real API spend"
      >
        <WriteGate enabled={canLaunch} reason={launchReason} testId="research-gate">
          <label className="opt-field">
            <span className="label">kind</span>
            <select value={kind} onChange={(e) => setKind(e.target.value)}>
              <option value="s7">s7</option>
              <option value="gepa_smoke">gepa_smoke</option>
            </select>
          </label>
          <label className="opt-field">
            <span className="label">cases</span>
            <input
              type="number"
              min={1}
              max={100}
              value={limit}
              onChange={(e) => setLimit(Number(e.target.value))}
            />
          </label>
          <label className="opt-field">
            <span className="label">retry_n</span>
            <input
              type="number"
              min={1}
              max={3}
              value={retryN}
              onChange={(e) => setRetryN(Number(e.target.value))}
            />
          </label>
          <button type="button" className="btn" data-testid="research-start" onClick={() => start.mutate()}>
            {start.isPending ? 'Starting…' : 'Start a round'}
          </button>
          <button type="button" className="linkbtn" data-testid="research-cancel" onClick={() => cancel.mutate()}>
            Cancel the running round
          </button>
        </WriteGate>

        {writeError && (
          <div className="callout err opt-msg" role="alert">
            <span className="ttl">✗ Refused</span>
            {writeError}
          </div>
        )}

        {status.isLoading ? (
          <LoadingRows rows={2} />
        ) : status.error ? (
          <ErrorNote error={status.error} />
        ) : status.data?.current ? (
          <div className="cards">
            <JobCard job={status.data.current} />
          </div>
        ) : (
          <EmptyState>
            No round is running. The harness is normally driven from the command line —{' '}
            <code>uv run python scripts/diagnose_failures.py --limit 10</code> — and this console is the
            same subprocess with its log tailed.
          </EmptyState>
        )}

        {history.length > 0 && (
          <>
            <h3>
              Previous rounds · {history.length}, {history.filter((j) => j.status === 'succeeded').length} succeeded
            </h3>
            <ul className="opt-jobs">
              {history.map((job) => (
                <li key={job.job_id}>
                  <span className="mono">{job.job_id}</span>
                  <span className="muted">{job.kind}</span>
                  <Verdict ok={job.status === 'succeeded'}>{job.status}</Verdict>
                  <span className="muted">{relativeTime(job.started_at)}</span>
                </li>
              ))}
            </ul>
          </>
        )}
      </Panel>
    </AdminPage>
  );
}
