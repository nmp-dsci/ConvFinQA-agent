import { BundleDiagram, PipelineDiagram } from './diagrams';
import {
  Callout,
  Field,
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
// 03 · Pipeline and routing
// ---------------------------------------------------------------------------

const STAGES: Array<{ stage: string; job: string; onNumber: string }> = [
  {
    stage: 'triage',
    job: 'Classifies the turn as a number look-up or a program, and the conversation as Type I or Type II.',
    onNumber: 'runs',
  },
  {
    stage: 'preprocess',
    job: 'Rewrites “this / that / the sum of both” into explicit sub-questions against the conversation’s own prior answers, and plans the program.',
    onNumber: 'skipped',
  },
  {
    stage: 'retriever',
    job: 'Answers each sub-question from the filing’s text and table, returning {question, answer} pairs.',
    onNumber: 'runs',
  },
  {
    stage: 'calculator',
    job: 'Reaches the answer through a loop over six arithmetic tools. The program is reconstructed from the tool trajectory afterwards.',
    onNumber: 'skipped',
  },
];

export function PipelineSection() {
  return (
    <Section
      id="pipeline"
      index={3}
      eyebrow="architecture"
      title="Four agents, and a router that decides how many of them run"
      lede={
        <>
          A turn is a question, the filing, and the conversation’s own prior answers — never the
          gold ones. Triage classifies it; a look-up takes a two-stage path and a computation takes
          a four-stage one. The whole turn is one async generator, so the events the browser
          streams and the events the recorded demo pack replays are emitted by the same code.
        </>
      }
    >
      <Panel>
        <PipelineDiagram />
      </Panel>

      <div>
        <ScrollX>
          <Table>
            <thead>
              <tr>
                <Th>stage</Th>
                <Th>what it does</Th>
                <Th>on a number turn</Th>
              </tr>
            </thead>
            <tbody>
              {STAGES.map((s) => (
                <tr key={s.stage}>
                  <Td>
                    <span className="type-num text-amber">{s.stage}</span>
                  </Td>
                  <Td>{s.job}</Td>
                  <Td>
                    <span
                      className={s.onNumber === 'runs' ? 'type-num text-good' : 'type-num text-faint'}
                    >
                      {s.onNumber}
                    </span>
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </ScrollX>
        <Provenance origin="code">
          <Mono>src/convfinqa/pipeline/runner.py :: turn_events()</Mono> — the single implementation.{' '}
          <Mono>run_turn()</Mono> drains it and <Mono>stream_turn()</Mono> forwards it, so there is
          no second code path to drift.
        </Provenance>
      </div>

      <Callout title="The routing decision is one string comparison, and that is worth knowing">
        <p>
          The branch is <Mono>if triage.turn_type == "number"</Mono>, where{' '}
          <Mono>turn_type</Mono> is a <Mono>Literal["number", "program"]</Mono> field the triage
          agent fills in. There is no heuristic, no confidence threshold and no fallback: a
          misclassified turn takes the wrong path entirely, and the trace shows which path it took.
          Conversation type is classified too but never routes anything — it is carried for slicing
          and for the preprocess input.
        </p>
      </Callout>

      <Panel>
        <PanelTitle>the calculator’s tools</PanelTitle>
        <Prose>
          Registered as plain tools on the calculator agent:{' '}
          <Mono>add</Mono> <Mono>subtract</Mono> <Mono>multiply</Mono> <Mono>divide</Mono>{' '}
          <Mono>exp</Mono> <Mono>greater</Mono>, each taking two floats. The agent never writes DSL
          text — it calls the operations, and the program string a reader sees in the inspector is
          rebuilt from the calls it actually made. That is why a program on this system is a record
          of what happened rather than a claim about it.
        </Prose>
        <Callout tone="warn" title="There is no cap on the tool loop">
          <p>
            No per-agent retry count, no usage limit and no maximum number of tool calls are set
            anywhere in the pipeline; the framework defaults apply. The only real bound on a stage
            is wall clock — <Mono>call_with_budget</Mono> abandons it at 120 seconds — plus the four
            transport attempts described in the next section. It has not caused a runaway in the
            evaluated runs, and it is still an unbounded loop with a timer around it rather than a
            bounded one.
          </p>
        </Callout>
      </Panel>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 04 · llm.py
// ---------------------------------------------------------------------------

const POLICY: Array<{ setting: string; value: string; why: string }> = [
  {
    setting: 'attempts',
    value: '4',
    why: 'Tenacity owns the retry; the HTTP client’s own retry is set to 0 so failures are counted once, not multiplied.',
  },
  {
    setting: 'backoff',
    value: 'exponential + jitter, 1s → 20s',
    why: 'Jitter matters because four agents in one turn fail at the same instant when a provider wobbles.',
  },
  {
    setting: 'per-call timeout',
    value: '120s (connect 10s)',
    why: 'A stage that has not answered in two minutes is abandoned as llm_unavailable rather than held open.',
  },
  {
    setting: 'retried on',
    value: '429, 5xx, timeouts, connect/read/protocol errors',
    why: 'These are the failures that go away on their own.',
  },
  {
    setting: 'never retried',
    value: '400, 401',
    why: 'A bad request repeated is the same bad request. Repeating it hides the cause and spends money.',
  },
];

export function LlmSection({ data }: { data: SystemData }) {
  const bundle = data.health?.bundle;
  return (
    <Section
      id="llm"
      index={4}
      eyebrow="the choke point"
      title="One module builds every model, and that is the whole point"
      lede={
        <>
          Two things have to be true of every model call in this system: it must be refused outright
          when the deployment is a keyless demo, and it must carry the same retry and timeout
          policy. Both live in <Mono>src/convfinqa/llm.py</Mono>. A model constructed anywhere else
          silently bypasses both, which is why the rule is a rule and not a preference.
        </>
      }
    >
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1.1fr)_minmax(0,1fr)]">
        <Panel>
          <PanelTitle>what the module owns</PanelTitle>
          <ul className="type-small space-y-2 text-muted">
            <li>
              <span className="type-num text-text">guard_llm_call()</span> — raises{' '}
              <Mono>DemoModeError</Mono> when the deployment is in demo mode. The route layer turns
              that into a 501 carrying <Mono>not_available_demo</Mono>, so a forged request fails at
              the server, not only in the UI.
            </li>
            <li>
              <span className="type-num text-text">get_provider() / get_model()</span> — the only
              construction site for the pydantic-ai model, wrapping a custom retrying transport.
            </li>
            <li>
              <span className="type-num text-text">model_settings()</span> — sends{' '}
              <Mono>extra_body</Mono> disabling the provider’s thinking mode, because the provider
              answers 400 to a request that pins a tool while thinking is on.
            </li>
            <li>
              <span className="type-num text-text">call_with_budget()</span> — the wall-clock ceiling
              every stage runs inside.
            </li>
          </ul>
          <Provenance origin="code">
            The backends expose <Mono>lm_mini()</Mono> / <Mono>lm_max()</Mono> factories that route
            through it — never a module-level model object, because importing a module must not
            require an API key.
          </Provenance>
        </Panel>

        <Panel>
          <PanelTitle>models this deployment is running</PanelTitle>
          <div className="space-y-3">
            <Field label="lm_mini · the four pipeline agents" note="one small fast model per stage">
              <Live value={bundle?.lm_mini} reason="/healthz did not answer" />
            </Field>
            <Field
              label="lm_max · diagnosis and optimisation"
              note="the s7 harness’s router and fix agents"
            >
              <Live value={bundle?.lm_max} reason="/healthz did not answer" />
            </Field>
            <Field label="mode" note="demo deployments hold no API key at all">
              <Live value={data.health?.mode} reason="/healthz did not answer" />
            </Field>
          </div>
        </Panel>
      </div>

      <div>
        <ScrollX>
          <Table>
            <thead>
              <tr>
                <Th>policy</Th>
                <Th numeric>value</Th>
                <Th>why it is that</Th>
              </tr>
            </thead>
            <tbody>
              {POLICY.map((p) => (
                <tr key={p.setting}>
                  <Td>{p.setting}</Td>
                  <Td numeric>{p.value}</Td>
                  <Td>{p.why}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </ScrollX>
        <Provenance origin="code">
          read from <Mono>src/convfinqa/llm.py</Mono> and <Mono>config.py::Settings</Mono>. These
          are the committed defaults, not a live read — the deployment does not expose them.
        </Provenance>
      </div>

      <Callout tone="broken" title="One path genuinely escapes this, and it is the broken one">
        <p>
          DSPy constructs its own client, so <Mono>backends/dspy.py::_lm()</Mono> builds a{' '}
          <Mono>dspy.LM</Mono> directly. It shares the demo gate and the single source of the API
          key, but not the retry transport — and, critically, not{' '}
          <Mono>model_settings()</Mono>. Neither it nor <Mono>dspy_lm_kwargs()</Mono> sends the
          thinking-mode <Mono>extra_body</Mono>, which is exactly why GEPA and every DSPy command
          still fails with the provider’s 400 today. See open work.
        </p>
      </Callout>

      <Callout title="Pinned by a test, because it broke the deployment twice">
        <p>
          Nothing may build a model at import time. Twice a read-only route returned a 500 purely
          because reading a dataset fact imported a module that constructed an LM in a container
          with no key.{' '}
          <Mono>tests/test_demo_mode.py::test_every_module_imports_without_a_key</Mono> imports
          every module with the environment stripped and fails the build if one of them reaches for
          a credential.
        </p>
      </Callout>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 05 · The bundle fingerprint
// ---------------------------------------------------------------------------

export function BundleSection({ data }: { data: SystemData }) {
  const bundle = data.health?.bundle;
  return (
    <Section
      id="bundle"
      index={5}
      eyebrow="versioning"
      title="What “a version” means when the model is somebody else’s API"
      lede={
        <>
          There are no weights to checkpoint here. What changes between versions is the prompts, an
          optional GEPA overlay, which models are called, which dataset was scored and which commit
          ran — so those six things are hashed together, and the twelve characters that come out are
          stamped on every recorded turn, every eval row and every registry entry.
        </>
      }
    >
      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Panel>
          <BundleDiagram bundleId={data.health?.bundle_id} />
        </Panel>

        <Panel>
          <PanelTitle>this deployment’s bundle</PanelTitle>
          <div className="space-y-2.5">
            <Field label="bundle_id">
              <Live value={data.health?.bundle_id} reason="/healthz did not answer" />
            </Field>
            <Field label="prompts_version" note="the generated prompt module in use">
              <Live value={bundle?.prompts_version} reason="/healthz did not answer" />
            </Field>
            <Field label="gepa_overlay" note="null unless an optimized runner is layered on">
              <Live value={bundle?.gepa_overlay ?? 'null'} reason="/healthz did not answer" />
            </Field>
            <Field label="dataset_hash" note="first 12 of sha256 over the dataset file">
              <Live value={bundle?.dataset_hash} reason="/healthz did not answer" />
            </Field>
            <Field
              label="code_sha"
              note="baked into the image at build time, or read from git in dev"
            >
              <Live value={bundle?.code_sha} reason="/healthz did not answer" />
            </Field>
          </div>
          <Provenance origin="live">
            read from <Mono>/healthz</Mono> on this deployment right now.
          </Provenance>
        </Panel>
      </div>

      <Prose>
        Because the id is derived rather than declared, it cannot be wrong: change a prompt and it
        moves, point at a different model and it moves, run the same code on a different dataset and
        it moves. The awkward consequence is honest too — a hosted model can change under a fixed
        id, so <Mono>lm_mini</Mono> pins the name the provider serves, not the weights behind it.
        This is a fingerprint of everything this repository controls, and it does not pretend to
        cover what it does not.
      </Prose>
    </Section>
  );
}
