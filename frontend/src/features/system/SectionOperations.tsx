import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { formatCount } from '../landing/format';
import { OPEN_WORK } from './benchmark';
import type { OpenStatus } from './benchmark';
import { DeployDiagram, ObservabilityDiagram } from './diagrams';
import {
  Callout,
  Field,
  Lamp,
  Live,
  Mono,
  Panel,
  PanelTitle,
  Prose,
  Provenance,
  ScrollX,
  Section,
  Table,
  Td,
  Th,
} from './ui';
import type { SystemData } from './useSystemData';

// ---------------------------------------------------------------------------
// 09 · Observability
// ---------------------------------------------------------------------------

interface SignalRow {
  signal: string;
  exists: string;
  visible: string;
  state: 'yes' | 'partial' | 'no';
}

const SIGNALS: SignalRow[] = [
  {
    signal: 'Per-stage input, output, reasoning and tool loop',
    exists: 'Recorded in the trace store on every turn',
    visible: 'The chat inspector, the traces list, and a deep link per turn',
    state: 'yes',
  },
  {
    signal: 'Answer against gold, per turn',
    exists: 'Recorded where gold exists',
    visible: 'Lamps, the accuracy tiles, the evaluations page',
    state: 'yes',
  },
  {
    signal: 'Errors and their classified cause',
    exists: 'Recorded as a code from a fixed enum, not free text',
    visible: 'The error-rate tile and the traces filter',
    state: 'yes',
  },
  {
    signal: 'Per-turn latency, tokens and cost',
    exists: 'Recorded for turns this process serves live',
    visible: 'Tiles and the inspector — but every recorded demo turn is unmeasured, so the demo shows em dashes',
    state: 'partial',
  },
  {
    signal: 'Per-model-call latency, retries, HTTP status, cache hits',
    exists: 'Constructed as OpenTelemetry spans on every call — and dropped',
    visible: 'Nowhere. Per-stage totals in the inspector are the closest this console gets',
    state: 'no',
  },
  {
    signal: 'HTTP route latency; rate-limit and in-flight rejections',
    exists: 'Route spans go to the same dropped exporter; the limiter counts nothing',
    visible: 'Nowhere in the app. The hosting platform’s own metrics are the source',
    state: 'no',
  },
  {
    signal: 'Deploy, uptime, container restarts',
    exists: 'The hosting platform only',
    visible: 'This page shows the build SHA and bundle; uptime is deliberately out of scope',
    state: 'partial',
  },
  {
    signal: 'Alerting',
    exists: 'A 5xx alarm on the hosted service. Nothing in the app',
    visible: 'Nowhere in the app',
    state: 'no',
  },
];

const STATE_STYLE: Record<SignalRow['state'], string> = {
  yes: 'text-good',
  partial: 'text-amber',
  no: 'text-bad',
};

const STATE_WORD: Record<SignalRow['state'], string> = {
  yes: 'visible',
  partial: 'partial',
  no: 'not captured',
};

export function ObservabilitySection({ data }: { data: SystemData }) {
  const captureEnabled = data.metrics?.trace_capture_enabled;
  const nTurns = data.metrics?.n_turns_total;

  return (
    <Section
      id="observability"
      index={9}
      eyebrow="observability"
      title="What is captured, what is not, and who decided"
      lede={
        <>
          Three layers exist. One of them reaches nothing, on purpose, and this section says so
          rather than listing “OpenTelemetry” as a feature. A visitor can see everything a turn did;
          a visitor cannot see what each individual model request did.
        </>
      }
    >
      <Panel>
        <ObservabilityDiagram />
      </Panel>

      <div className="grid gap-3 lg:grid-cols-3">
        <Panel>
          <div className="mb-1.5 flex items-center gap-2">
            <Lamp state="replay" />
            <span className="mono-caps">1 · OpenTelemetry spans</span>
          </div>
          <Prose className="type-small">
            Every HTTP request, every agent run, every model request with its tokens, every tool
            call and every turn is wrapped in a span through the Logfire SDK. Export is configured
            as <Mono>if-token-present</Mono>, and no token is set locally, in the container image,
            in the infrastructure or in CI. The spans are built on every turn and thrown away.
          </Prose>
          <p className="type-meta mt-2 text-faint">Who can see this today: nobody.</p>
        </Panel>

        <Panel>
          <div className="mb-1.5 flex items-center gap-2">
            <Lamp state="good" />
            <span className="mono-caps">2 · the per-turn trace store</span>
          </div>
          <Prose className="type-small">
            One SQLite row per turn: each stage’s input, output, reasoning and tool trajectory,
            plus latency, tokens, cost, the answer, the gold answer, whether it was right, the
            bundle id, any error code, and a <Mono>source</Mono> of serving, demo or eval so live
            and replayed turns are never summed.
          </Prose>
          <div className="mt-2 space-y-1">
            <p className="type-meta">
              capture{' '}
              <span className={captureEnabled === false ? 'text-bad' : 'text-good'}>
                <Live
                  value={
                    captureEnabled === undefined
                      ? null
                      : captureEnabled
                        ? 'enabled'
                        : 'disabled'
                  }
                  reason="/metrics/production did not answer"
                />
              </span>
              {typeof nTurns === 'number' && (
                <>
                  {' '}
                  · <span className="type-num">{formatCount(nTurns)}</span> turns held
                </>
              )}
            </p>
            <p className="type-meta text-faint">
              Everyone can see this — it is the layer the decision kept.
            </p>
          </div>
        </Panel>

        <Panel>
          <div className="mb-1.5 flex items-center gap-2">
            <Lamp state="good" />
            <span className="mono-caps">3 · experiment tracking</span>
          </div>
          <Prose className="type-small">
            Evaluation, GEPA and s7 runs with their metrics and artefacts, champion and challenger
            aliases, and an append-only promotion history. Live from MLflow in development; the
            demo image reads a committed export of the same thing, so the history survives a
            container with no database.
          </Prose>
          <p className="type-meta mt-2 text-faint">Everyone can see this.</p>
        </Panel>
      </div>

      <Callout tone="warn" title="The decision, dated: turns, not spans">
        <p>
          On 29 August the scope was set to per-turn capture only. No span processor and no service
          counters were built, and the traces surface keeps its name rather than becoming a
          three-tab observability page. The trade is stated rather than hidden: a reader gets{' '}
          <em>what the system did</em> — every turn, every stage, every tool call, its correctness —
          but not <em>what each model request did</em>. That becomes the wrong trade the day this
          serves real traffic and someone has to answer “why was that turn slow”.
        </p>
        <p>
          The recovery path costs nothing to keep open and is not hypothetical: the spans already
          exist in this process, so setting <Mono>LOGFIRE_TOKEN</Mono> in a development shell sends
          the full request, agent and model-call tree with no code change at all.
        </p>
      </Callout>

      <div>
        <ScrollX>
          <Table>
            <thead>
              <tr>
                <Th>signal</Th>
                <Th>what exists</Th>
                <Th>where a visitor sees it</Th>
                <Th>state</Th>
              </tr>
            </thead>
            <tbody>
              {SIGNALS.map((row) => (
                <tr key={row.signal}>
                  <Td className="text-text">{row.signal}</Td>
                  <Td>{row.exists}</Td>
                  <Td>{row.visible}</Td>
                  <Td>
                    <span className={cn('type-num whitespace-nowrap', STATE_STYLE[row.state])}>
                      {STATE_WORD[row.state]}
                    </span>
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </ScrollX>
        <Provenance origin="code">
          verified against <Mono>serving/app.py</Mono>, <Mono>tracking/traces.py</Mono> and{' '}
          <Mono>serving/routes/metrics.py</Mono>.{' '}
          <Link
            to="/admin/traces"
            className="text-amber underline decoration-amber-line underline-offset-4"
          >
            Open the turns →
          </Link>
        </Provenance>
      </div>

      <Callout tone="broken" title="In the public demo, every timing figure is an em dash">
        <p>
          The recorded pack holds eight filings and fifty-five turns, and every stage event in it
          carries an empty metrics object — no latency, no token counts. So the demo deployment’s
          latency, token and cost surfaces print <Mono>—</Mono> with the reason, rather than a
          zero or a flat line. Two things make that the right rendering rather than a gap: a turn
          recorded at 6.7 s and replayed in 2 s is a 6.7-second turn, so replay timing is not
          latency in the first place; and an unmeasured turn shown as a measured zero is a lie in
          the flattering direction. Filling them in needs a metered evaluation run and a re-record,
          which costs real API calls and has not been authorised.
        </p>
      </Callout>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 10 · Deploy
// ---------------------------------------------------------------------------

const GATE_LAYERS = [
  {
    layer: 'the route table',
    detail:
      'Write routes are not registered at all when the deployment is a demo. There is nothing to call.',
  },
  {
    layer: 'the interface',
    detail:
      'Every write control sits inside a real disabled fieldset with the reason shown — not a pointer-events trick, and never simply hidden. A visitor sees the whole console, including what it refuses.',
  },
  {
    layer: 'the server',
    detail:
      'A forged write returns 403 or 501 regardless of what the client believes. The deployment smoke test asserts exactly this by posting a promotion and requiring a refusal.',
  },
];

export function DeploySection({ data }: { data: SystemData }) {
  return (
    <Section
      id="deploy"
      index={10}
      eyebrow="deployment"
      title="One image, two deployments, and no way for infrastructure to make it billable"
      lede={
        <>
          The same container runs in development with a key and in public without one. The switch is
          an environment line in the image, not a setting in the infrastructure — because a
          deployment variable that can be flipped is a deployment variable that will be.
        </>
      }
    >
      <Panel>
        <DeployDiagram />
      </Panel>

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Panel>
          <PanelTitle>the demo gate — three layers, all three required</PanelTitle>
          <ol className="type-small ml-4 list-decimal space-y-2 text-muted marker:text-faint">
            {GATE_LAYERS.map((l) => (
              <li key={l.layer}>
                <span className="text-text">{l.layer}</span> — {l.detail}
              </li>
            ))}
          </ol>
          <Prose className="mt-3">
            The public container holds no API key at all, so even a bug that got past all three
            would have nothing to spend.
          </Prose>
        </Panel>

        <Panel>
          <PanelTitle>this deployment, right now</PanelTitle>
          <div className="space-y-2.5">
            <Field label="mode" note="demo deployments are keyless and replay recorded turns">
              <span className="inline-flex items-center gap-2">
                <Lamp state={data.isDemo ? 'replay' : 'good'} />
                <Live value={data.health?.mode} reason="/healthz did not answer" />
              </span>
            </Field>
            <Field label="champion" note="the version answering questions here">
              <Live value={data.health?.champion} reason="/healthz did not answer" />
            </Field>
            <Field label="code_sha" note="baked in at image build time">
              <Live value={data.health?.bundle.code_sha} reason="/healthz did not answer" />
            </Field>
            <Field label="recorded filings available" note="what the demo can replay without a key">
              <Live
                value={
                  data.health ? String(data.health.demo_reports) : null
                }
                reason="/healthz did not answer"
              />
            </Field>
          </div>
          <Provenance origin="live">
            <Mono>/healthz</Mono> — the same endpoint the platform’s health check polls.
          </Provenance>
        </Panel>
      </div>

      <Panel>
        <PanelTitle>the pipeline from commit to public URL</PanelTitle>
        <ul className="type-small space-y-2 text-muted">
          <li>
            GitHub Actions assumes an AWS role over OIDC. There is no stored access key in the
            repository or in the workflow — the credential is minted per run and expires with it.
          </li>
          <li>
            The image is built for <Mono>linux/amd64</Mono> with the commit SHA passed in as a build
            argument, so the running container can tell you which commit it is without being asked
            to trust a tag. It is pushed to a private registry under both{' '}
            <Mono>:latest</Mono> and <Mono>:&lt;sha&gt;</Mono>; rollback is retagging a previous SHA.
          </li>
          <li>
            The service is applied by Terraform and polled until it reports running, with a health
            check on <Mono>/healthz</Mono>. The image configuration block contains only the port —
            no environment variables and no secret references, deliberately.
          </li>
          <li>
            A smoke script then asserts six things against the live URL, including that a promotion
            posted to the public deployment is refused.
          </li>
        </ul>
        <Provenance origin="code">
          <Mono>.github/workflows/deploy-aws.yml</Mono>, <Mono>Dockerfile</Mono>,{' '}
          <Mono>infra/terraform/demo/</Mono>, <Mono>scripts/demo_smoke.sh</Mono>.
        </Provenance>
      </Panel>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 11 · Make a new version
// ---------------------------------------------------------------------------

const STEPS: Array<{ title: string; body: string; cmd?: string }> = [
  {
    title: 'Change the prompts, and let the harness write them',
    body: 'Run the s7 loop over the failing cases, or edit the per-agent rule stores directly. Then regenerate the variant module — it is generated from the JSONL stores and never hand-edited, which is what keeps a prompt traceable to the case that caused it.',
    cmd: 'uv run python scripts/diagnose_failures.py --limit 10 --retry-n 3\nuv run python scripts/diagnose_failures.py --stage assemble',
  },
  {
    title: 'Score it, and commit the CSV',
    body: 'Run the evaluation to produce the prediction CSVs for the new version, then commit them. Cached evaluations are tracked in git on purpose: it is how v1, v2 and v3_1 reproduce on another machine with no API calls at all, and it is what every number on this site is computed from.',
    cmd: 'uv run convfinqa-eval\ngit add evaluation/predictions/pydantic_predictions_<v>*.csv',
  },
  {
    title: 'Compare it against the champion',
    body: 'The comparator joins the two versions on the questions they both answered and reports the accuracy delta and every pass→fail flip. It exits non-zero when the candidate is not promotable, so this is usable as a check rather than only as a report.',
    cmd: 'uv run convfinqa-mlflow compare <champion> <v>',
  },
  {
    title: 'Promote, and export what the demo reads',
    body: 'Promotion is refused unless the comparator passes; forcing it is possible and is recorded as forced in the append-only history. Then export the snapshot, because the demo image has no MLflow database and reads the committed file.',
    cmd: 'uv run convfinqa-mlflow promote <v>\nuv run convfinqa-mlflow snapshot',
  },
  {
    title: 'Re-record the demo pack and ship',
    body: 'Rebuild the recorded pack from the committed CSVs so the public demo replays the new champion, then push. CI runs the eval gate; on success the image is built, pushed and applied, and the smoke script checks the live URL.',
    cmd: 'uv run convfinqa-demo-pack --n 8\ngit push origin main',
  },
];

export function NewVersionSection() {
  return (
    <Section
      id="new-version"
      index={11}
      eyebrow="operating it"
      title="Make a new version, in five steps"
      lede={
        <>
          Nothing here is a wrapper around a hidden process. Each step is a command, each command
          leaves an artefact in the repository, and the artefacts are what the pages in this console
          read.
        </>
      }
    >
      <ol className="space-y-2.5">
        {STEPS.map((step, i) => (
          <li key={step.title} className="min-w-0 rounded-md border border-line bg-panel p-3.5">
            <div className="flex items-baseline gap-2.5">
              <span className="type-num shrink-0 text-amber">{i + 1}</span>
              <div className="min-w-0 flex-1">
                <p className="type-body font-medium text-text">{step.title}</p>
                <p className="type-small mt-1 max-w-[68ch] text-muted">{step.body}</p>
                {step.cmd && (
                  <pre className="mt-2 overflow-x-auto rounded-sm border border-line-2 bg-ground p-2 text-[11px] leading-relaxed text-muted">
                    {step.cmd}
                  </pre>
                )}
              </div>
            </div>
          </li>
        ))}
      </ol>
      <Callout tone="warn" title="Step 1 has a caveat while GEPA is broken">
        <p>
          The s7 harness runs on the pydantic-ai path and works. GEPA does not — the DSPy client
          does not send the provider’s thinking-mode override and every run fails with a 400 before
          it starts. Until that is fixed, “change the prompts” means the s7 harness or a human, not
          the optimizer.
        </p>
      </Callout>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 12 · Open work
// ---------------------------------------------------------------------------

const STATUS_LABEL: Record<OpenStatus, string> = {
  broken: 'broken today',
  open: 'open',
  deferred: 'deliberately deferred',
};

const STATUS_CLASS: Record<OpenStatus, string> = {
  broken: 'border-bad/55 text-bad',
  open: 'border-amber-line text-amber',
  deferred: 'border-line-2 text-faint',
};

export function OpenWorkSection({ data }: { data: SystemData }) {
  const broken = OPEN_WORK.filter((i) => i.status === 'broken');
  const rest = OPEN_WORK.filter((i) => i.status !== 'broken');
  const ordered = [...broken, ...rest];

  return (
    <Section
      id="open-work"
      index={12}
      eyebrow="open work"
      title="What is broken, what is unfinished, and what was left out on purpose"
      lede={
        <>
          Two of the items below do not work right now. They are first, they are labelled broken,
          and they say what does not work as a result. A debrief that lists only future work is a
          roadmap, and this page is worth reading only if it is true.
        </>
      }
    >
      <ul className="space-y-2.5">
        {ordered.map((item) => (
          <li key={item.title} className="min-w-0 rounded-md border border-line bg-panel p-3.5">
            <div className="flex flex-wrap items-baseline justify-between gap-x-3 gap-y-1">
              <span className="type-body font-medium text-text">{item.title}</span>
              <span
                className={cn(
                  'mono-caps shrink-0 rounded-sm border px-1.5 py-px',
                  STATUS_CLASS[item.status],
                )}
              >
                {STATUS_LABEL[item.status]}
              </span>
            </div>
            <p className="type-small mt-1.5 max-w-[72ch] text-muted">{item.body}</p>
            {item.where && (
              <p className="type-num mt-1.5 text-[10.5px] text-faint">{item.where}</p>
            )}
          </li>
        ))}
      </ul>

      {data.failures.length > 0 && (
        <Callout tone="note" title="Live reads that did not answer while this page was loaded">
          <p>
            <span className="type-num">{data.failures.join(' · ')}</span>. Everything static on this
            page still stands; the live values above print an em dash rather than a remembered
            number. If the backend is not running, that is the expected appearance.
          </p>
        </Callout>
      )}
    </Section>
  );
}
