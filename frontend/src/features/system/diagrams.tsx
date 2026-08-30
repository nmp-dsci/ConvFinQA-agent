import type { ReactNode } from 'react';

/**
 * Four hand-authored diagrams.
 *
 * All of them are inline SVG with a 420-unit viewBox and a 600px cap, so at a
 * 420px phone they render close to 1:1 and on a wide column they scale up
 * rather than stranding a fixed-width picture in white space. No diagramming
 * dependency and no raster export: a diagram that cannot follow the theme is a
 * diagram that is wrong half the time, and every colour below is a token.
 *
 * The shared vocabulary, kept consistent across all four:
 *   solid outline   — real, running, reachable
 *   dashed outline  — real code that reaches nothing, or a deliberate refusal
 *   amber fill      — the choke point / the thing the section is about
 */

const UI = 'var(--font-ui)';
const MONO = 'var(--font-mono)';

function Figure({
  children,
  caption,
  label,
  height,
}: {
  children: ReactNode;
  caption: ReactNode;
  label: string;
  height: number;
}) {
  return (
    <figure className="m-0 min-w-0">
      <svg
        viewBox={`0 0 420 ${height}`}
        role="img"
        aria-label={label}
        className="block h-auto w-full max-w-[600px]"
      >
        <defs>
          <marker
            id="sysArrow"
            viewBox="0 0 10 10"
            refX="9"
            refY="5"
            markerWidth="6"
            markerHeight="6"
            orient="auto-start-reverse"
          >
            <path d="M0 0 L10 5 L0 10 z" fill="var(--faint)" />
          </marker>
        </defs>
        {children}
      </svg>
      <figcaption className="type-meta mt-2 max-w-[62ch]">{caption}</figcaption>
    </figure>
  );
}

function Box({
  x,
  y,
  w,
  h,
  dashed = false,
  accent = false,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  dashed?: boolean;
  accent?: boolean;
}) {
  return (
    <rect
      x={x}
      y={y}
      width={w}
      height={h}
      rx="4"
      fill={accent ? 'var(--amber-soft)' : 'var(--panel)'}
      stroke={accent ? 'var(--amber-line)' : 'var(--line-2)'}
      strokeWidth="1"
      strokeDasharray={dashed ? '4 3' : undefined}
      opacity={dashed ? 0.75 : 1}
    />
  );
}

function T({
  x,
  y,
  children,
  size = 10.5,
  fill = 'var(--text)',
  anchor = 'middle',
  mono = false,
  weight = 400,
  opacity = 1,
}: {
  x: number;
  y: number;
  children: ReactNode;
  size?: number;
  fill?: string;
  anchor?: 'start' | 'middle' | 'end';
  mono?: boolean;
  weight?: number;
  opacity?: number;
}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      fontSize={size}
      fontWeight={weight}
      fill={fill}
      fontFamily={mono ? MONO : UI}
      opacity={opacity}
    >
      {children}
    </text>
  );
}

function Arrow({
  x1,
  y1,
  x2,
  y2,
  dashed = false,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  dashed?: boolean;
}) {
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke="var(--faint)"
      strokeWidth="1"
      strokeDasharray={dashed ? '4 3' : undefined}
      markerEnd="url(#sysArrow)"
      opacity={dashed ? 0.6 : 0.9}
    />
  );
}

// ---------------------------------------------------------------------------
// 1 · The pipeline
// ---------------------------------------------------------------------------

export function PipelineDiagram() {
  return (
    <Figure
      height={286}
      label="The four-agent pipeline. A turn enters as a question plus the conversation's own prior answers. Triage classifies it as a number look-up or a program. A number turn runs two stages, triage then retriever, and returns the retrieved value with an empty program. A program turn runs four, adding preprocess before the retriever and a calculator afterwards that reaches the answer through a loop over six arithmetic tools."
      caption={
        <>
          Two paths, chosen by one string. Everything below the branch is skipped on a look-up —
          that is where the 87.7% on number questions comes from, and it is visible in the
          inspector as stages that never started.
        </>
      }
    >
      <Box x={8} y={8} w={404} h={30} />
      <T x={210} y={22} weight={500}>
        one turn
      </T>
      <T x={210} y={33} size={8.5} fill="var(--muted)">
        the question · the document · the conversation’s own prior answers
      </T>

      <Arrow x1={210} y1={38} x2={210} y2={54} />

      <Box x={116} y={56} w={188} h={32} accent />
      <T x={210} y={70} weight={500}>
        triage
      </T>
      <T x={210} y={82} size={8.5} mono fill="var(--muted)">
        turn_type ∈ &#123;number, program&#125;
      </T>

      {/* Branch */}
      <path
        d="M210 88 L210 98 M210 98 L102 98 L102 110 M210 98 L318 98 L318 110"
        fill="none"
        stroke="var(--faint)"
        strokeWidth="1"
        opacity="0.9"
      />
      {/* Beside the drop lines, not on them — a label centred on x=102 would
          have the connector running straight through its middle. */}
      <T x={96} y={108} size={8.5} mono fill="var(--good)" anchor="end">
        number · 2 stages
      </T>
      <T x={324} y={108} size={8.5} mono fill="var(--info)" anchor="start">
        program · 4 stages
      </T>

      {/* Left path */}
      <Box x={12} y={114} w={180} h={32} />
      <T x={102} y={128} weight={500}>
        retriever
      </T>
      <T x={102} y={139} size={8.5} fill="var(--muted)">
        returns &#123;question, answer&#125; pairs
      </T>

      <Box x={12} y={158} w={180} h={44} dashed />
      <T x={102} y={176} size={9.5} fill="var(--faint)">
        preprocess and calculator
      </T>
      <T x={102} y={190} size={9.5} fill="var(--faint)">
        never run
      </T>

      {/* Right path */}
      <Box x={228} y={114} w={180} h={32} />
      <T x={318} y={128} weight={500}>
        preprocess
      </T>
      <T x={318} y={139} size={8.5} fill="var(--muted)">
        “this” → an explicit sub-question
      </T>

      <Arrow x1={318} y1={146} x2={318} y2={156} />

      <Box x={228} y={158} w={180} h={30} />
      <T x={318} y={172} weight={500}>
        retriever
      </T>
      <T x={318} y={183} size={8.5} fill="var(--muted)">
        one lookup per sub-question
      </T>

      <Arrow x1={318} y1={188} x2={318} y2={198} />

      <Box x={228} y={200} w={180} h={44} accent />
      <T x={318} y={214} weight={500}>
        calculator
      </T>
      <T x={318} y={226} size={8.5} mono fill="var(--muted)">
        add subtract multiply
      </T>
      <T x={318} y={237} size={8.5} mono fill="var(--muted)">
        divide exp greater
      </T>

      <path
        d="M102 202 L102 256 M318 244 L318 256 M102 256 L318 256"
        fill="none"
        stroke="var(--faint)"
        strokeWidth="1"
        opacity="0.9"
      />
      <Arrow x1={210} y1={256} x2={210} y2={264} />

      <Box x={80} y={266} w={260} h={16} />
      <T x={210} y={277} size={9.5} fill="var(--muted)">
        answer + reconstructed program, recorded as one turn
      </T>
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
  return (
    <Figure
      height={168}
      label="The bundle fingerprint: six fields — prompts version, GEPA overlay, both model ids, dataset hash and code SHA — are serialised as canonical JSON, hashed with SHA-256, and truncated to twelve hex characters to form the bundle id that every recorded turn carries."
      caption={
        <>
          Six fields in, twelve hex characters out. Every recorded turn, every eval row and every
          registry entry carries this id, which is what makes “which version answered that?” a
          question with an answer when the model is somebody else’s API.
        </>
      }
    >
      {BUNDLE_FIELDS.map((field, i) => {
        const col = i % 2;
        const row = Math.floor(i / 2);
        const x = 8 + col * 132;
        const y = 8 + row * 26;
        return (
          <g key={field}>
            <Box x={x} y={y} w={124} h={20} />
            <T x={x + 62} y={y + 14} size={9} mono fill="var(--muted)">
              {field}
            </T>
          </g>
        );
      })}

      <path
        d="M272 44 L296 44 L296 84"
        fill="none"
        stroke="var(--faint)"
        strokeWidth="1"
        opacity="0.9"
      />
      <Arrow x1={296} y1={80} x2={296} y2={92} />
      <T x={330} y={40} size={8.5} mono fill="var(--faint)" anchor="start">
        sorted keys,
      </T>
      <T x={330} y={51} size={8.5} mono fill="var(--faint)" anchor="start">
        no whitespace
      </T>

      <Box x={206} y={94} w={180} h={26} />
      <T x={296} y={110} size={9.5} mono fill="var(--muted)">
        sha256(json)[:12]
      </T>

      <Arrow x1={296} y1={120} x2={296} y2={130} />
      <Box x={206} y={132} w={180} h={28} accent />
      <T x={296} y={150} size={11} mono weight={500} fill="var(--amber)">
        {bundleId ?? 'bundle_id'}
      </T>
    </Figure>
  );
}

// ---------------------------------------------------------------------------
// 3 · Observability
// ---------------------------------------------------------------------------

export function ObservabilityDiagram() {
  return (
    <Figure
      height={210}
      label="Two paths leave every turn. The upper path constructs OpenTelemetry spans through the Logfire SDK on every request, agent run, model call and tool call; they are exported only when a Logfire token is present, and no token exists locally or in the demo image, so they are dropped. The lower path records one row per turn into a SQLite trace store, which the traces routes and the production metrics endpoint read, and which is what this console shows."
      caption={
        <>
          The dashed path is real code that reaches nothing. It is drawn dashed rather than left
          out because it is the recovery path: the spans already exist in this process, and
          setting <code>LOGFIRE_TOKEN</code> in a dev shell turns the whole tree on with no code
          change.
        </>
      }
    >
      <Box x={8} y={82} w={104} h={46} />
      <T x={60} y={100} weight={500}>
        one turn
      </T>
      <T x={60} y={112} size={8.5} fill="var(--muted)">
        4 agents · ≤ 8 model
      </T>
      <T x={60} y={122} size={8.5} fill="var(--muted)">
        calls · 1 HTTP request
      </T>

      <Arrow x1={112} y1={96} x2={148} y2={48} dashed />
      <Arrow x1={112} y1={114} x2={148} y2={158} />

      {/* Span path — dashed, goes nowhere */}
      <Box x={150} y={16} w={148} h={52} dashed />
      <T x={224} y={32} size={10} fill="var(--muted)" weight={500}>
        Logfire SDK spans
      </T>
      <T x={224} y={44} size={8.5} fill="var(--faint)">
        request · agent run
      </T>
      <T x={224} y={55} size={8.5} fill="var(--faint)">
        model call · tool call
      </T>
      <T x={224} y={65} size={8} mono fill="var(--faint)">
        constructed every turn
      </T>

      <Arrow x1={298} y1={42} x2={330} y2={42} dashed />
      <Box x={332} y={22} w={80} h={40} dashed />
      <T x={372} y={38} size={9} fill="var(--faint)">
        Logfire cloud
      </T>
      <T x={372} y={50} size={8} mono fill="var(--bad)">
        no token — dropped
      </T>

      {/* Trace store path — solid */}
      <Box x={150} y={140} w={148} h={52} />
      <T x={224} y={156} size={10} weight={500}>
        TraceStore · SQLite
      </T>
      <T x={224} y={168} size={8.5} fill="var(--muted)">
        one row per turn: stages,
      </T>
      <T x={224} y={179} size={8.5} fill="var(--muted)">
        latency, tokens, gold, error
      </T>
      <T x={224} y={189} size={7.5} mono fill="var(--faint)">
        source ∈ serving | demo | eval
      </T>

      <Arrow x1={298} y1={166} x2={310} y2={166} />
      <Box x={312} y={140} w={100} h={52} accent />
      <T x={362} y={158} size={9} weight={500}>
        this console
      </T>
      <T x={362} y={170} size={8} mono fill="var(--muted)">
        /traces
      </T>
      <T x={362} y={181} size={7.5} mono fill="var(--muted)">
        /metrics/production
      </T>
    </Figure>
  );
}

// ---------------------------------------------------------------------------
// 4 · Deploy
// ---------------------------------------------------------------------------

export function DeployDiagram() {
  // A straight vertical chain rather than a wrapped grid: a two-row layout
  // needs a connector from the end of one row back to the start of the next,
  // and at this width that line spans the whole figure and reads as a stray
  // rule under the first row rather than as a step.
  const stops: Array<[string, string]> = [
    ['push to main', 'GitHub'],
    ['CI, including the eval gate', 'tests · re-score every committed CSV'],
    ['docker buildx · linux/amd64', 'the commit SHA baked in as CODE_SHA'],
    ['push to the private registry', 'tagged :latest and :<sha>'],
    ['terraform apply', 'App Runner · health check on /healthz'],
    ['smoke the live URL', '6 assertions · a write must be refused'],
  ];
  return (
    <Figure
      height={286}
      label="Deployment: a push to main runs CI including the eval regression gate, then a linux/amd64 image is built with the code SHA baked in, pushed to ECR under both a latest and a commit tag, applied to App Runner by Terraform, and finally checked by a six-assertion smoke script against the live URL."
      caption={
        <>
          The credential story is the point: GitHub assumes an AWS role over OIDC, so there is no
          stored access key anywhere in the repository, and <code>DEMO_MODE=1</code> is an{' '}
          <code>ENV</code> line in the Dockerfile rather than a Terraform variable — infrastructure
          is not able to turn the public URL into a billable one.
        </>
      }
    >
      {stops.map(([title, sub], i) => {
        const y = 8 + i * 44;
        return (
          <g key={title}>
            <Box x={40} y={y} w={340} h={34} accent={i === 1} />
            <T x={210} y={y + 15} size={10} weight={500}>
              {title}
            </T>
            <T x={210} y={y + 27} size={8.5} mono fill="var(--muted)">
              {sub}
            </T>
            {i < stops.length - 1 && <Arrow x1={210} y1={y + 34} x2={210} y2={y + 43} />}
          </g>
        );
      })}

      <T x={210} y={278} size={8.5} mono fill="var(--faint)">
        no stored AWS keys · OIDC role assumption · DEMO_MODE baked into the image
      </T>
    </Figure>
  );
}
