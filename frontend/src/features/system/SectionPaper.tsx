import { Link } from 'react-router-dom';
import { Figure } from '@/kit/ui';
import { versionLabel } from '../admin/lib';
import { ARMS } from '../admin/runtimeStory';
import type { RuntimeKey } from '../admin/runtimeStory';
import { NO_VALUE } from '../landing/format';
import { DistributionBars, PerTurnChart, SliceChart, SliceLegend } from './charts';
import type { TurnPoint } from './charts';
import { sliceRows } from './benchmark';
import { Dia } from './diagrams';
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
import type { BaselineRow } from './paper';
import {
  Callout,
  Cite,
  Field,
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
import './system.css';

// ---------------------------------------------------------------------------
// The leaderboard rows: the paper's systems and ours, on one axis
// ---------------------------------------------------------------------------

export interface BoardRow {
  key: string;
  /** The full name, for the table. */
  system: string;
  /** The figure's row label. */
  short: string;
  family: BaselineRow['family'] | 'ours';
  trained: string;
  /** Execution accuracy in points, or `null` when it was never measured. */
  exe: number | null;
  exeText: string;
  progText: string;
  evaluatedOn: string;
  source: string;
  ours: boolean;
  ceiling: boolean;
}

function points(value: string | null): number | null {
  const n = value === null ? NaN : Number.parseFloat(value);
  return Number.isFinite(n) ? n : null;
}

function pct(value: number | null | undefined): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value * 100 : null;
}

const OUR_ORDER: RuntimeKey[] = ['agent_sdk', 'pipeline'];

/**
 * The paper's rows plus one per runtime arm, sorted on execution accuracy. Our rows come from
 * `/eval/campaigns` only; an arm that has not run keeps its row, prints why, and sorts last.
 */
export function boardRows(data: SystemData): BoardRow[] {
  const paper: BoardRow[] = BASELINES.map((row) => ({
    key: row.system,
    system: row.system,
    short: row.short,
    family: row.family,
    trained: row.trained,
    exe: points(row.exe),
    exeText: row.exe ?? NO_VALUE,
    progText: row.prog ?? NO_VALUE,
    evaluatedOn: row.evaluatedOn,
    source: row.source,
    ours: false,
    ceiling: Boolean(row.ceiling),
  }));

  const reason =
    data.runtimes === undefined
      ? '/eval/campaigns did not answer'
      : 'this arm has not been scored on the gate split';

  const ours: BoardRow[] = OUR_ORDER.map((key) => {
    const arm = data.runtimes?.[key] ?? null;
    const exe = pct(arm?.accuracy);
    const prog = pct(arm?.program_accuracy);
    const version = versionLabel(arm?.version) || (key === 'agent_sdk' ? 'sdk champion' : 'champion');
    return {
      key: `ours-${key}`,
      system: `${version} · ${ARMS[key].title}${arm?.model ? ` (${arm.model})` : ''}`,
      short: `${version} · ${ARMS[key].title}`,
      family: 'ours',
      trained: 'prompts only · no weights trained',
      exe,
      exeText: exe === null ? `not scored — ${reason}` : exe.toFixed(1),
      progText: prog === null ? NO_VALUE : prog.toFixed(1),
      evaluatedOn:
        data.gateQuestions !== null
          ? `${data.gateQuestions}-question gate split${data.gateReports !== null ? ` · ${data.gateReports} filings` : ''}`
          : 'the gate split',
      source: '/eval/campaigns',
      ours: true,
      ceiling: false,
    };
  });

  return [...paper, ...ours].sort((a, b) => (b.exe ?? -1) - (a.exe ?? -1));
}

// ---------------------------------------------------------------------------
// The figure: one bar per system, the human expert as the reference line
// ---------------------------------------------------------------------------

function BoardFigure({ rows }: { rows: BoardRow[] }) {
  const ceiling = rows.find((r) => r.ceiling);
  const bars = rows.filter((r) => !r.ceiling);
  const W = 1100;
  const x0 = 400;
  const span = 600;
  const top = 44;
  const rowH = 30;
  const H = top + bars.length * rowH + 52;
  const xOf = (v: number) => x0 + (v / 100) * span;

  return (
    <Figure
      testId="leaderboard-figure"
      title="fig 1 · execution accuracy, every system on one axis"
      caption={
        <>
          Our two runtimes in the accent colour, the paper&rsquo;s systems in grey, the paper&rsquo;s
          human expert as the vertical rule. Read the caveat under the figure before the gaps.
        </>
      }
      source={
        <>
          {PAPER.authors}, {PAPER.venue}, Tables 5, 6 and 9; ours from{' '}
          <code className="lit">/eval/campaigns</code>
        </>
      }
    >
      <Dia
        id="board"
        w={W}
        h={H}
        label={`Execution accuracy for ${bars.length} systems on a 0 to 100 axis, with the human expert at ${ceiling?.exeText ?? 'the paper figure'} as a reference line. ${bars
          .map((r) => `${r.short}: ${r.exe === null ? 'not scored' : r.exeText}`)
          .join('; ')}.`}
      >
        {[0, 25, 50, 75, 100].map((tick) => (
          <g key={tick}>
            <line className="gl" x1={xOf(tick)} x2={xOf(tick)} y1={top - 6} y2={H - 44} />
            <text className="tx k" x={xOf(tick)} y={H - 24} textAnchor="middle">
              {tick}
            </text>
          </g>
        ))}
        <text className="tx s" x={x0 + span} y={H - 4} textAnchor="end">
          execution accuracy, % of questions
        </text>

        {bars.map((row, i) => {
          const y = top + i * rowH;
          const cls = row.ours ? ' hi' : '';
          return (
            <g key={row.key} className="pt">
              <title>{`${row.system}: ${row.exe === null ? row.exeText : `${row.exeText}%`}`}</title>
              <text className={`tx${cls}`} x={x0 - 12} y={y + 20} textAnchor="end">
                {row.short}
              </text>
              {row.exe !== null && (
                <rect className={`bar${cls}`} x={x0} y={y + 7} width={(row.exe / 100) * span} height={16} rx="2" />
              )}
              <text
                className={`tx k${cls}`}
                x={row.exe === null ? x0 + 8 : xOf(row.exe) + 8}
                y={y + 20}
              >
                {row.exeText}
              </text>
            </g>
          );
        })}

        {ceiling?.exe != null && (
          <g>
            <line className="ax mark" x1={xOf(ceiling.exe)} x2={xOf(ceiling.exe)} y1={top - 10} y2={H - 44} strokeDasharray="5 4" />
            <text className="tx k" x={xOf(ceiling.exe)} y={24} textAnchor="middle">
              human expert · {ceiling.exeText}
            </text>
          </g>
        )}
      </Dia>
    </Figure>
  );
}

function caveat(title: RegExp) {
  return BENCHMARK_CAVEATS.find((c) => title.test(c.title));
}

// ---------------------------------------------------------------------------
// 01 · The leaderboard
// ---------------------------------------------------------------------------

export function LeaderboardSection({ data }: { data: SystemData }) {
  const rows = boardRows(data);
  const era = caveat(/different era/i);
  const questions = caveat(/different questions/i);
  const rest = BENCHMARK_CAVEATS.filter((c) => c !== era && c !== questions);
  let rank = 0;

  return (
    <Section
      id="leaderboard"
      index={1}
      eyebrow="leaderboard"
      title="Every published system and ours, on execution accuracy"
      lede={
        <>
          The paper&rsquo;s side is fixed and cited. Ours is read live from{' '}
          <Mono>/eval/campaigns</Mono> — both runtime champions, scored on the same unseen gate
          split — so promoting a challenger moves the row. The caveats are part of the figure.
        </>
      }
    >
      <BoardFigure rows={rows} />

      <Callout tone="warn" title="Different paradigm, different questions, possibly seen before">
        {era && <p>{era.body}</p>}
        {questions && <p>{questions.body}</p>}
        <p>
          Those public files have been on GitHub since 2022, so a hosted model trained later may
          have seen them. Nothing here rules that out.
        </p>
      </Callout>

      <div>
        <ScrollX>
          <Table>
            <thead>
              <tr>
                <Th numeric>#</Th>
                <Th>system</Th>
                <Th numeric>exe acc</Th>
                <Th numeric>prog acc</Th>
                <Th>evaluated on</Th>
                <Th>source</Th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                if (!row.ceiling && row.exe !== null) rank += 1;
                return (
                  <tr key={row.key} className={row.ours ? 'pro' : undefined} data-ours={row.ours || undefined}>
                    <Td numeric className="lb-rank">
                      {row.ceiling ? 'ref' : row.exe === null ? NO_VALUE : rank}
                    </Td>
                    <td className="sub lb-sys">
                      {row.system}
                      <span className="path">{row.ceiling ? 'two annotators, over 85% agreement — the reference line' : row.trained}</span>
                    </td>
                    <Td numeric>{row.exe === null ? NO_VALUE : row.exeText}</Td>
                    <Td numeric>{row.progText}</Td>
                    <Td>{row.exe === null && row.ours ? row.exeText : row.evaluatedOn}</Td>
                    <Td>{row.ours ? <span className="tag acc">{row.exe === null ? 'ours · not read' : 'ours · live'}</span> : <Cite>{row.source}</Cite>}</Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        </ScrollX>
        <Provenance origin="paper">
          every row but ours · Tables 5, 6 and 9. Our rows are read live; when the backend is not
          running they print why, never a remembered number.
        </Provenance>
      </div>

      <div className="sys-two">
        {rest.map((c) => (
          <Callout key={c.title} tone={/program accuracy/i.test(c.title) ? 'warn' : 'note'} title={c.title}>
            <p>{c.body}</p>
          </Callout>
        ))}
      </div>
    </Section>
  );
}

// ---------------------------------------------------------------------------
// 02 · By slice and by turn
// ---------------------------------------------------------------------------

export function BenchmarkSection({ data }: { data: SystemData }) {
  const rows = sliceRows(data.championSlices);
  const champion = versionLabel(data.championName);

  // The paper's Figure 5 is 1-based over turns; our slices are 0-based q_order straight out of
  // the predictions CSV. The chart does the shift.
  const perTurn: TurnPoint[] = (data.championSlices?.by_q_order ?? [])
    .map((s) => ({ order: Number(s.label), accuracy: s.accuracy * 100, n: s.n_total }))
    .filter((p) => Number.isFinite(p.order))
    .sort((a, b) => a.order - b.order);

  return (
    <Section
      id="benchmark"
      index={2}
      eyebrow="by slice"
      title="Hybrid conversations and later turns cost every system"
      lede={
        <>
          The paper&rsquo;s cuts, with the pipeline champion
          {champion ? (
            <>
              {' '}
              (<Mono>{champion}</Mono>)
            </>
          ) : null}{' '}
          beside its two best baselines. Hybrid conversations concatenate two multi-hop questions,
          so dependency chains are longer; a later turn inherits every error before it.
        </>
      }
    >
      <Panel>
        <PanelTitle>accuracy by question type and conversation type</PanelTitle>
        <SliceLegend />
        <SliceChart rows={rows} />
        <Provenance origin="committed">
          our bars are recomputed on request from{' '}
          <Mono>evaluation/predictions/pydantic_predictions_&lt;version&gt;_joined.csv</Mono> — no
          API calls. The paper&rsquo;s are its best fine-tuned model and best prompted GPT-3 setting{' '}
          <Cite>Table 4 / Table 6</Cite>.
        </Provenance>
      </Panel>

      <div className="sys-two">
        <Figure
          testId="perturn-figure"
          title="fig 2 · accuracy against turn position"
          caption="A turn depends on the answers before it, so a curve falling to the right is error compounding, not later questions being harder."
          source={<Cite>Figure 5</Cite>}
        >
          <PerTurnChart ours={perTurn} paper={PER_TURN_BASELINE} />
        </Figure>
        <div>
          <PanelTitle>why this curve is the paper&rsquo;s real finding</PanelTitle>
          <Prose>
            FinQANet falls from <span className="num">75.6</span> at the first turn to{' '}
            <span className="num">34.4</span> at the sixth; GPT-3 from <span className="num">72.8</span>{' '}
            to <span className="num">25.2</span> <Cite>Figure 5</Cite>. Once a turn is wrong, the
            turns that depend on it are almost certainly wrong too.
          </Prose>
          <Prose>
            This pipeline threads the conversation&rsquo;s own prior answers rather than gold, so it
            inherits the same compounding. Only the two endpoints of Figure 5 are transcribed here,
            so the paper&rsquo;s line is drawn between them rather than invented into a curve.
          </Prose>
        </div>
      </div>

      <div>
        <PanelTitle>the paper&rsquo;s findings, and what this system does about each</PanelTitle>
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
                  <Td className="wrap">{finding.paper}</Td>
                  <Td className="wrap">{finding.response}</Td>
                  <Td className="wrap">
                    {finding.evidence}
                    {finding.to && (
                      <>
                        <br />
                        <Link to={finding.to} className="mono">
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

// ---------------------------------------------------------------------------
// 03 · The paper
// ---------------------------------------------------------------------------

export function PaperSection() {
  return (
    <Section
      id="paper"
      index={3}
      eyebrow="the task"
      title="Chained questions over filings; a human expert reaches 89.44%"
      lede={
        <>
          The benchmark is <cite>{PAPER.title}</cite> — {PAPER.authors}, {PAPER.venue}. Given a
          filing&rsquo;s text and table, answer a sequence of questions where later ones depend on
          earlier ones. Two annotators with over 85% agreement reach{' '}
          <span className="num">89.44%</span> execution accuracy on a 200-question sample.
        </>
      }
    >
      <Panel>
        <PanelTitle>the corpus</PanelTitle>
        <div className="sys-fields cols">
          {DATASET_SCALE.map((stat) => (
            <Field
              key={stat.label}
              label={stat.label}
              note={
                <>
                  {stat.note && <>{stat.note} </>}
                  <Cite>{stat.source}</Cite>
                </>
              }
            >
              <span className="num">{stat.value}</span>
            </Field>
          ))}
        </div>
        <Provenance origin="paper">
          arXiv {PAPER.arxivId}, committed at <Mono>{PAPER.localPath}</Mono> so the citation is
          checkable without leaving the repository.
        </Provenance>
      </Panel>

      <div className="sys-two">
        {DISTRIBUTIONS.map((block) => (
          <Panel key={block.title}>
            <div className="panel-head">
              <h3>{block.title}</h3>
              <span className="right">
                <Cite>{block.source}</Cite>
              </span>
            </div>
            <p className="note">{block.caption}</p>
            <DistributionBars bars={block.bars} />
          </Panel>
        ))}
      </div>

      <div>
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
                  <Td className="sub mono">{op.op}</Td>
                  <Td className="mono">{op.args}</Td>
                  <Td>{op.out}</Td>
                  <Td className="mono">{op.meaning}</Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </ScrollX>
        <Prose>
          Intermediate results are referenced positionally — <Mono>#0</Mono>, <Mono>#1</Mono> — so a
          two-step answer is written <Mono>subtract(243, 111), divide(#0, 111)</Mono>. This pipeline
          never writes that string: its calculator calls the six operations as tools and the program
          is rebuilt from the tool loop, the format the paper found models handle best — and the
          root of the program-accuracy gap in the leaderboard.
        </Prose>
      </div>
    </Section>
  );
}
