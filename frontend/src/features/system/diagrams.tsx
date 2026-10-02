import type { ReactNode } from 'react';
import { Figure } from '@/kit/ui';

/**
 * Four hand-authored diagrams, on the kit's `.dia` vocabulary.
 *
 * Every one is inline SVG on an 1100-unit viewBox — the width of the page column — so at desktop
 * width a `.tx` label renders at its 13px and never below the house floor of 12.8px. On a narrower
 * screen the figure keeps that minimum width and scrolls inside its own wrapper rather than shrinking
 * its text (`Dia`). No colour is written here: `.nd`, `.ed`, `.tx` read the tokens.
 *
 *   `.nd`       — real, running, reachable
 *   `.nd.ext`   — real code that reaches nothing, or a stage that never starts
 *   `.nd.hi`    — the choke point: the thing the section is about
 */

/** The narrowest a figure may render, so a 13-unit label never lands under 12.8px. */
function minWidthFor(w: number): number {
  return Math.ceil((w * 12.8) / 13);
}

/** A `.dia` SVG in its own scroll wrapper, with a per-figure arrow marker. */
export function Dia({
  id,
  w,
  h,
  label,
  children,
}: {
  id: string;
  w: number;
  h: number;
  label: string;
  children: ReactNode;
}) {
  return (
    <div className="figscroll">
      <svg
        className="dia"
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label={label}
        style={{ minWidth: minWidthFor(w) }}
      >
        <defs>
          <marker
            id={`arr-${id}`}
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="7"
            markerHeight="7"
            orient="auto-start-reverse"
          >
            <path d="M0 0L10 5L0 10z" fill="currentColor" />
          </marker>
        </defs>
        {children}
      </svg>
    </div>
  );
}

/** A node: the box, then its lines — the first as the title, the rest as `s` (sans) or `k` (mono). */
function Node({
  x,
  y,
  w,
  h,
  tone,
  title,
  lines = [],
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  tone?: 'hi' | 'ext';
  title?: ReactNode;
  lines?: Array<[string, ReactNode]>;
}) {
  const start = title ? y + 28 : y + h / 2 + 5 - ((lines.length - 1) * 21) / 2;
  return (
    <g>
      <rect className={`nd${tone ? ` ${tone}` : ''}`} x={x} y={y} width={w} height={h} rx="8" />
      {title && (
        <text className={`tx${tone === 'ext' ? ' s' : ''}`} x={x + 16} y={start}>
          {title}
        </text>
      )}
      {lines.map(([kind, text], i) => (
        <text
          key={i}
          className={`tx ${kind}`}
          x={x + 16}
          y={title ? start + 23 + i * 21 : start + i * 21}
        >
          {text}
        </text>
      ))}
    </g>
  );
}

function Edge({ id, d, dash = false }: { id: string; d: string; dash?: boolean }) {
  return <path className={`ed${dash ? ' dash' : ''}`} d={d} markerEnd={`url(#arr-${id})`} />;
}

function Line({ d }: { d: string }) {
  return <path className="ed" d={d} />;
}

// ---------------------------------------------------------------------------
// 1 · The pipeline
// ---------------------------------------------------------------------------

export function PipelineDiagram() {
  const id = 'pipeline';
  return (
    <Figure
      testId="pipeline-figure"
      title="fig 1 · one turn, routed by one string"
      caption={
        <>
          Two paths, chosen by triage&rsquo;s <code className="lit">turn_type</code>. A look-up runs
          two stages and a computation four; on a number turn the inspector shows preprocess and the
          calculator as stages that never started.
        </>
      }
      source={<code className="lit">src/convfinqa/pipeline/runner.py :: turn_events()</code>}
    >
      <Dia
        id={id}
        w={1100}
        h={364}
        label="The four-agent pipeline. A turn enters as a question plus the conversation's own prior answers. Triage classifies it as a number look-up or a program. A number turn runs two stages, triage then retriever, and returns the retrieved value with an empty program. A program turn runs four, adding preprocess before the retriever and a calculator afterwards that reaches the answer through a loop over six arithmetic tools. Both paths end in one recorded turn: the answer and the reconstructed program."
      >
        <Node
          x={16}
          y={94}
          w={170}
          h={96}
          title="one turn"
          lines={[
            ['s', 'question · filing'],
            ['s', '+ its prior answers'],
          ]}
        />
        <Edge id={id} d="M186 142 L208 142" />
        <Node
          x={210}
          y={94}
          w={160}
          h={96}
          tone="hi"
          title="triage"
          lines={[
            ['k', 'turn_type ∈'],
            ['k', '{number,program}'],
          ]}
        />

        {/* The branch: one string comparison, two rows. */}
        <Line d="M370 142 L404 142 M404 64 L404 220" />
        <Edge id={id} d="M404 64 L428 64" />
        <Edge id={id} d="M404 220 L428 220" />
        <text className="tx k" x={430} y={26}>
          number · 2 stages
        </text>
        <text className="tx k" x={430} y={162}>
          program · 4 stages
        </text>

        {/* Number row: the retriever, then straight to the answer. */}
        <Node
          x={430}
          y={36}
          w={240}
          h={56}
          title="retriever"
          lines={[['s', '{question, answer} pairs']]}
        />
        <text className="tx s" x={700} y={56}>
          preprocess and calculator never start
        </text>
        <Edge id={id} d="M670 64 L1092 64 L1092 324 L1086 324" />

        {/* Program row: four stages. */}
        <Node
          x={430}
          y={172}
          w={200}
          h={96}
          title="preprocess"
          lines={[
            ['s', '“this” → an explicit'],
            ['s', 'sub-question'],
          ]}
        />
        <Edge id={id} d="M630 220 L652 220" />
        <Node
          x={654}
          y={172}
          w={180}
          h={96}
          title="retriever"
          lines={[
            ['s', 'one lookup per'],
            ['s', 'sub-question'],
          ]}
        />
        <Edge id={id} d="M834 220 L856 220" />
        <Node
          x={858}
          y={172}
          w={220}
          h={96}
          tone="hi"
          title="calculator"
          lines={[
            ['k', 'add subtract multiply'],
            ['k', 'divide exp greater'],
          ]}
        />
        <Edge id={id} d="M968 268 L968 298" />

        <Node
          x={430}
          y={300}
          w={654}
          h={48}
          lines={[['s', 'answer + reconstructed program, recorded as one turn']]}
        />
      </Dia>
    </Figure>
  );
}

// ---------------------------------------------------------------------------
// 2 · The bundle fingerprint
// ---------------------------------------------------------------------------

const BUNDLE_FIELDS = [
  'prompts_version',
  'gepa_overlay',
  'lm_mini',
  'lm_max',
  'dataset_hash',
  'code_sha',
];

export function BundleDiagram({ bundleId }: { bundleId?: string }) {
  const id = 'bundle';
  return (
    <Figure
      testId="bundle-figure"
      title="fig 2 · six fields in, twelve hex characters out"
      caption={
        <>
          Every recorded turn, every eval row and every registry entry carries this id, which is what
          makes &ldquo;which version answered that?&rdquo; a question with an answer when the model
          is somebody else&rsquo;s API.
        </>
      }
      source={<code className="lit">src/convfinqa/tracking/bundle.py</code>}
    >
      <Dia
        id={id}
        w={1100}
        h={180}
        label="The bundle fingerprint: six fields — prompts version, GEPA overlay, both model ids, dataset hash and code SHA — are serialised as canonical JSON, hashed with SHA-256, and truncated to twelve hex characters to form the bundle id that every recorded turn carries."
      >
        {BUNDLE_FIELDS.map((field, i) => {
          const x = 16 + (i % 3) * 216;
          const y = 28 + Math.floor(i / 3) * 64;
          return (
            <g key={field}>
              <rect className="nd" x={x} y={y} width={200} height={44} rx="8" />
              <text className="tx k" x={x + 16} y={y + 27}>
                {field}
              </text>
            </g>
          );
        })}
        <text className="tx s" x={16} y={166}>
          serialised as canonical JSON — sorted keys, no whitespace
        </text>
        <Edge id={id} d="M648 82 L698 82" />
        <Node
          x={700}
          y={50}
          w={190}
          h={64}
          title="hash"
          lines={[['k', 'sha256(json)[:12]']]}
        />
        <Edge id={id} d="M890 82 L916 82" />
        <rect className="nd hi" x={918} y={50} width={166} height={64} rx="8" />
        <text className="tx" x={934} y={78}>
          bundle_id
        </text>
        <text className="tx k hi" x={934} y={101}>
          {bundleId ?? 'not read'}
        </text>
      </Dia>
    </Figure>
  );
}

// ---------------------------------------------------------------------------
// 3 · Observability
// ---------------------------------------------------------------------------

export function ObservabilityDiagram() {
  const id = 'obs';
  return (
    <Figure
      testId="observability-figure"
      title="fig 3 · two paths leave every turn, and one reaches nothing"
      caption={
        <>
          The dashed path is real code that reaches nothing. It is drawn rather than left out because
          it is the recovery path: the spans already exist in this process, and setting{' '}
          <code className="lit">LOGFIRE_TOKEN</code> in a dev shell turns the whole tree on with no
          code change.
        </>
      }
      source={
        <>
          <code className="lit">serving/app.py</code>, <code className="lit">tracking/traces.py</code>
        </>
      }
    >
      <Dia
        id={id}
        w={1100}
        h={260}
        label="Two paths leave every turn. The upper path constructs OpenTelemetry spans through the Logfire SDK on every request, agent run, model call and tool call; they are exported only when a Logfire token is present, and no token exists locally or in the demo image, so they are dropped. The lower path records one row per turn into a SQLite trace store, which the traces routes and the production metrics endpoint read, and which is what this app shows."
      >
        <Node
          x={16}
          y={82}
          w={200}
          h={96}
          title="one turn"
          lines={[
            ['s', '4 agents · ≤ 8 model'],
            ['s', 'calls · 1 HTTP request'],
          ]}
        />
        <Edge id={id} dash d="M216 110 L246 110 L246 60 L278 60" />
        <Edge id={id} d="M216 150 L246 150 L246 200 L278 200" />

        <Node
          x={280}
          y={16}
          w={340}
          h={88}
          tone="ext"
          title="Logfire SDK spans"
          lines={[
            ['s', 'request · agent run · model call · tool'],
            ['k', 'constructed every turn'],
          ]}
        />
        <Edge id={id} dash d="M620 60 L698 60" />
        <Node
          x={700}
          y={16}
          w={384}
          h={88}
          tone="ext"
          title="Logfire cloud"
          lines={[['k', 'no token — dropped']]}
        />

        <Node
          x={280}
          y={156}
          w={340}
          h={88}
          title="TraceStore · SQLite"
          lines={[
            ['s', 'one row per turn: stages, latency, gold'],
            ['k', 'source ∈ serving | demo | eval'],
          ]}
        />
        <Edge id={id} d="M620 200 L698 200" />
        <Node
          x={700}
          y={156}
          w={384}
          h={88}
          tone="hi"
          title="this app"
          lines={[['k', '/traces · /metrics/production']]}
        />
      </Dia>
    </Figure>
  );
}

// ---------------------------------------------------------------------------
// 4 · Deploy
// ---------------------------------------------------------------------------

const STOPS: Array<{ title: string; sub: string[] }> = [
  { title: 'push to main', sub: ['GitHub'] },
  { title: 'CI + eval gate', sub: ['tests · re-score', 'every committed CSV'] },
  { title: 'docker buildx', sub: ['linux/amd64 ·', 'CODE_SHA baked in'] },
  { title: 'push to registry', sub: ['tagged :latest', 'and :<sha>'] },
  { title: 'terraform apply', sub: ['App Runner ·', 'health on /healthz'] },
  { title: 'smoke the URL', sub: ['6 assertions · a', 'write is refused'] },
];

export function DeployDiagram() {
  const id = 'deploy';
  return (
    <Figure
      testId="deploy-figure"
      title="fig 4 · commit to public URL in six stops"
      caption={
        <>
          GitHub assumes an AWS role over OIDC, so no access key is stored anywhere, and{' '}
          <code className="lit">DEMO_MODE=1</code> is an <code className="lit">ENV</code> line in the
          Dockerfile rather than a Terraform variable: infrastructure cannot turn the public URL into
          a billable one.
        </>
      }
      source={<code className="lit">.github/workflows/deploy-aws.yml</code>}
    >
      <Dia
        id={id}
        w={1100}
        h={170}
        label="Deployment: a push to main runs CI including the eval regression gate, then a linux/amd64 image is built with the code SHA baked in, pushed to ECR under both a latest and a commit tag, applied to App Runner by Terraform, and finally checked by a six-assertion smoke script against the live URL."
      >
        {STOPS.map((stop, i) => {
          const x = 18 + i * 180;
          return (
            <g key={stop.title}>
              <Node
                x={x}
                y={20}
                w={164}
                h={100}
                tone={i === 1 ? 'hi' : undefined}
                title={stop.title}
                lines={stop.sub.map((s) => ['s', s] as [string, string])}
              />
              {i < STOPS.length - 1 && <Edge id={id} d={`M${x + 164} 70 L${x + 178} 70`} />}
            </g>
          );
        })}
        <text className="tx k" x={18} y={154}>
          no stored AWS keys · OIDC role assumption · DEMO_MODE baked into the image
        </text>
      </Dia>
    </Figure>
  );
}
