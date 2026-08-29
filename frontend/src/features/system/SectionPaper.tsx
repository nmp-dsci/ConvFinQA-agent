import { Link } from 'react-router-dom';
import { cn } from '@/lib/utils';
import { formatPercent, NO_VALUE } from '../landing/format';
import { PerTurnChart, DistributionBars, SliceChart, SliceLegend } from './charts';
import type { TurnPoint } from './charts';
import { sliceRows } from './benchmark';
import {
  BASELINES,
  BENCHMARK_CAVEATS,
  DATASET_SCALE,
  DISTRIBUTIONS,
  DSL_OPS,
  FINDINGS,
  PAPER,
  PER_TURN_BASELINE,
} from './paper';
import {
  Callout,
  Cite,
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
// 01 · The paper
// ---------------------------------------------------------------------------

export function PaperSection() {
  return (
    <Section
      id="paper"
      index={1}
      eyebrow="the task"
      title="What ConvFinQA asks, and how far a human gets"
      lede={
        <>
          The benchmark is <em>{PAPER.title}</em> — {PAPER.authors}, {PAPER.venue}. Given a filing’s
          text and table, answer a sequence of questions where later questions depend on earlier
          ones, and produce a program that can be executed to get there. Two annotators with more
          than 85% agreement reach <span className="type-num text-good">89.44%</span> execution
          accuracy on a 200-question sample. That is the ceiling everything on this page is measured
          against.
        </>
      }
    >
      <Panel>
        <PanelTitle>the corpus</PanelTitle>
        <div className="grid gap-x-6 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
          {DATASET_SCALE.map((stat) => (
            <Field
              key={stat.label}
              label={stat.label}
              note={
                <span className="inline-flex flex-wrap items-baseline gap-1.5">
                  {stat.note}
                  <Cite>{stat.source}</Cite>
                </span>
              }
            >
              <span className="type-num">{stat.value}</span>
            </Field>
          ))}
        </div>
        <Provenance origin="paper">
          arXiv {PAPER.arxivId}, committed at <Mono>{PAPER.localPath}</Mono> so the citation is
          checkable without leaving the repository.
        </Provenance>
      </Panel>

      <div className="grid gap-3 md:grid-cols-2">
        {DISTRIBUTIONS.map((block) => (
          <Panel key={block.title}>
            <div className="mb-1.5 flex items-baseline justify-between gap-2">
              <span className="mono-caps">{block.title}</span>
              <Cite>{block.source}</Cite>
            </div>
            <p className="type-meta mb-2.5">{block.caption}</p>
            <DistributionBars bars={block.bars} />
          </Panel>
        ))}
      </div>

      <Panel>
        <PanelTitle>the six-operation DSL the gold programs are written in</PanelTitle>
        <ScrollX>
          <Table>
            <thead>
              <tr>
                <Th>op</Th>
                <Th>arguments</Th>
                <Th>returns</Th>
                <Th>meaning</Th>
              </tr>
            </thead>
            <tbody>
              {DSL_OPS.map((op) => (
                <tr key={op.op}>
                  <Td>
                    <span className="type-num text-amber">{op.op}</span>
                  </Td>
                  <Td>
                    <span className="type-num">{op.args}</span>
                  </Td>
                  <Td>{op.out}</Td>
                  <Td>
                    <span className="type-num">{op.meaning}</span>
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </ScrollX>
        <Prose className="mt-3">
          Intermediate results are referenced positionally — <Mono>#0</Mono>, <Mono>#1</Mono> — so a
          two-step answer is written{' '}
          <Mono>subtract(243, 111), divide(#0, 111)</Mono>. This pipeline never writes that string.
          Its calculator calls the six operations as tools and the program is reconstructed from the
          tool loop afterwards, which is the format the paper found models handle best — and which
          is also the root of the program-accuracy gap two sections down.
        </Prose>
      </Panel>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 02 · The benchmark
// ---------------------------------------------------------------------------

function OurRow({ data }: { data: SystemData }) {
  const { championVersion, championHoldout, championName } = data;
  return (
    <tr className="bg-amber-soft">
      <Td className="text-text">
        <strong className="font-medium">This deployment</strong>
        {championName && (
          <>
            {' · '}
            <span className="type-num text-amber">{championName}</span>
          </>
        )}
        <div className="type-meta mt-0.5">
          four agents on a hosted LLM · prompts tuned by GEPA and the s7 harness · no weights
          trained
        </div>
      </Td>
      <Td>120 conversations, prompts only</Td>
      <Td numeric>
        <Live
          value={
            championVersion ? formatPercent(championVersion.exe_acc, 2).replace('%', '') : null
          }
          reason="/admin/versions did not answer"
        />
        <div className="type-meta mt-0.5 whitespace-nowrap font-normal">
          <Live
            value={
              championHoldout
                ? formatPercent(championHoldout.holdout_accuracy, 2).replace('%', '')
                : null
            }
            reason="/admin/experiments did not answer"
            className="text-good"
          />{' '}
          never-seen
        </div>
      </Td>
      <Td numeric>
        <Live
          value={
            championVersion ? formatPercent(championVersion.prog_acc, 2).replace('%', '') : null
          }
          reason="/admin/versions did not answer"
        />
        <div className="type-meta mt-0.5 whitespace-nowrap font-normal">see caveat 4</div>
      </Td>
      <Td>
        <Live
          value={championVersion ? String(championVersion.n_questions) : null}
          reason="unknown without /admin/versions"
        />{' '}
        q / 200 conv. from the public split
      </Td>
      <Td>
        {/* The badge reports whether this row was actually read, not merely
            that it is *meant* to be live — a "live" label over a row of em
            dashes would be the exact kind of decoration this page argues
            against. */}
        {championVersion ? (
          <span className="mono-caps rounded-sm border border-good-line px-1 py-px text-good">
            live
          </span>
        ) : (
          <span className="mono-caps rounded-sm border border-line-2 px-1 py-px text-faint">
            not read
          </span>
        )}
      </Td>
    </tr>
  );
}

export function BenchmarkSection({ data }: { data: SystemData }) {
  const rows = sliceRows(data.championSlices);

  // The paper's Figure 5 is 1-based over turns; our slices are 0-based q_order
  // straight out of the predictions CSV. The chart does the shift, so nothing
  // here has to remember which convention it is holding.
  const perTurn: TurnPoint[] = (data.championSlices?.by_q_order ?? [])
    .map((s) => ({ order: Number(s.label), accuracy: s.accuracy * 100, n: s.n_total }))
    .filter((p) => Number.isFinite(p.order))
    .sort((a, b) => a.order - b.order);

  return (
    <Section
      id="benchmark"
      index={2}
      eyebrow="the benchmark"
      title="Where this system lands among the paper’s baselines"
      lede={
        <>
          The paper’s side of this table is fixed and cited. This deployment’s side is read live
          from <Mono>/admin/versions</Mono> and <Mono>/admin/experiments</Mono>, so promoting a
          challenger moves the row and nobody has to remember to edit a paragraph. The four caveats
          below the table are part of the table.
        </>
      }
    >
      <div>
        <ScrollX>
          <Table>
            <thead>
              <tr>
                <Th>system</Th>
                <Th>trained / tuned on</Th>
                <Th numeric>exe acc</Th>
                <Th numeric>prog acc</Th>
                <Th>evaluated on</Th>
                <Th>source</Th>
              </tr>
            </thead>
            <tbody>
              {BASELINES.map((row, i) => [
                // Our row is spliced in directly under the human ceiling, which
                // is where it belongs on the leaderboard and — more usefully —
                // is where a reader will compare it against the line nothing
                // here beats rather than against the systems it edges out.
                i === 1 ? <OurRow key="ours" data={data} /> : null,
                <tr key={row.system}>
                  <Td className={cn(row.ceiling && 'text-text')}>
                    {row.ceiling ? (
                      <strong className="font-medium">{row.system}</strong>
                    ) : (
                      row.system
                    )}
                    {row.ceiling && (
                      <div className="type-meta mt-0.5">
                        two annotators, over 85% agreement — the ceiling
                      </div>
                    )}
                  </Td>
                  <Td>{row.trained}</Td>
                  <Td numeric className={cn(row.ceiling && 'text-good')}>
                    {row.exe ?? NO_VALUE}
                  </Td>
                  <Td numeric className={cn(row.ceiling && 'text-good')}>
                    {row.prog ?? NO_VALUE}
                  </Td>
                  <Td>{row.evaluatedOn}</Td>
                  <Td>
                    <Cite>{row.source}</Cite>
                  </Td>
                </tr>,
              ])}
            </tbody>
          </Table>
        </ScrollX>
        <Provenance origin="paper">
          every row but ours · <span className="type-num">Tables 5, 6 and 9</span>. Our row is read
          live; when the backend is not running it prints em dashes rather than a remembered
          number.
        </Provenance>
      </div>

      <div className="grid gap-2.5 md:grid-cols-2">
        {BENCHMARK_CAVEATS.map((caveat, i) => (
          <Callout
            key={caveat.title}
            tone={i === 3 ? 'warn' : 'note'}
            title={`${i + 1} · ${caveat.title}`}
          >
            <p>{caveat.body}</p>
          </Callout>
        ))}
      </div>

      <Panel>
        <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
          <span className="mono-caps">by question type and conversation type</span>
          <SliceLegend />
        </div>
        <SliceChart rows={rows} />
        <Provenance origin="committed">
          our bars are recomputed on request from{' '}
          <Mono>evaluation/predictions/pydantic_predictions_&lt;version&gt;_joined.csv</Mono> — no API
          calls, nothing cached to drift. The paper’s are its best fine-tuned model and its best
          prompted GPT-3 setting <Cite>Table 4 / Table 6</Cite>. The dashed green rule is the human
          ceiling at 89.44.
        </Provenance>
      </Panel>

      <div className="grid gap-3 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Panel>
          <PanelTitle>accuracy against turn position</PanelTitle>
          <PerTurnChart ours={perTurn} paper={PER_TURN_BASELINE} />
        </Panel>
        <Panel>
          <PanelTitle>why this curve is the paper’s real finding</PanelTitle>
          <Prose>
            FinQANet falls from <span className="type-num">75.6</span> at the first turn to{' '}
            <span className="type-num">34.4</span> at the sixth; GPT-3 from{' '}
            <span className="type-num">72.8</span> to <span className="type-num">25.2</span>{' '}
            <Cite>Figure 5</Cite>. Errors compound: once a turn is wrong, the turns that depend on
            it are almost certainly wrong too.
          </Prose>
          <Prose className="mt-2.5">
            This pipeline threads the conversation’s own prior answers rather than gold, so it
            inherits the same compounding — and the curve is a first-class slice rather than a
            footnote, because the overall figure hides it. Only the two endpoints of Figure 5 are
            transcribed in this repo, so the paper’s line is drawn between them and labelled as
            such rather than invented into a curve.
          </Prose>
        </Panel>
      </div>

      <div>
        <PanelTitle>the paper’s findings, and what this system does about each</PanelTitle>
        <ScrollX>
          <Table>
            <thead>
              <tr>
                <Th>what the paper found</Th>
                <Th>what this system does</Th>
                <Th>where to check it</Th>
              </tr>
            </thead>
            <tbody>
              {FINDINGS.map((finding) => (
                <tr key={finding.paper}>
                  <Td>{finding.paper}</Td>
                  <Td className="text-text">{finding.response}</Td>
                  <Td>
                    {finding.evidence}
                    {finding.to && (
                      <>
                        <br />
                        <Link
                          to={finding.to}
                          className="type-num text-amber underline decoration-amber-line underline-offset-4 hover:decoration-amber"
                        >
                          {finding.to}
                        </Link>
                      </>
                    )}
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </ScrollX>
        <Provenance origin="paper">
          the left column is quoted from §5.3 and §6.3; the middle and right describe this
          repository.
        </Provenance>
      </div>
    </Section>
  );
}
