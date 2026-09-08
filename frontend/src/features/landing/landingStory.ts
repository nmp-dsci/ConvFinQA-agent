import type { CampaignsResponse } from '../admin/api';
import { PAPER_HUMAN, runtimeVerdict } from '../admin/runtimeStory';
import { NO_VALUE, formatPercent, formatPointsDelta } from './format';

/**
 * What the landing says, as pure functions over `/eval/campaigns`.
 *
 * The headline is the strongest claim the record supports and nothing more:
 * "human-expert accuracy" is only written when the single-session arm has
 * been scored on the gate split and its figure is at or above the paper's
 * published human figure — and even then it travels with the caveat. Every
 * branch has a test, the rule the s13 review forced on the judge tile.
 */

export interface ProofStat {
  key: 'runtime' | 'human' | 'delta';
  value: string;
  label: string;
  tone: 'text' | 'violet' | 'good' | 'bad' | 'faint';
}

export interface LandingStory {
  /** The H1. `emphasis` is the one phrase set in amber, or null. */
  headline: { before: string; emphasis: string | null; after: string };
  lede: string;
  proof: ProofStat[];
  /** The caveat that must sit beside the human comparison, or null. */
  caveat: string | null;
  /** The sealed-holdout sentence under the tiles. */
  holdout: string;
  measured: boolean;
}

function ratioOrNull(v: number | null | undefined): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

export function landingStory(
  campaigns: CampaignsResponse | undefined,
  isDemo: boolean,
): LandingStory {
  const sdk = ratioOrNull(campaigns?.runtime_comparison?.agent_sdk?.accuracy);
  const human = ratioOrNull(PAPER_HUMAN.exe);
  const verdict = runtimeVerdict(campaigns?.runtime_comparison, campaigns?.split);
  const questions = verdict.gateQuestions;
  const reports = verdict.gateReports;
  const measured = sdk !== null;
  const atHuman = measured && human !== null && sdk >= human;

  const headline = !measured
    ? {
        before: 'A system that answers ',
        emphasis: 'dependent',
        after: ' questions about SEC filings — and shows its work.',
      }
    : atHuman
      ? {
          before: 'Answers dependent questions about SEC filings at ',
          emphasis: 'human-expert',
          after: ' accuracy — and shows every step.',
        }
      : {
          before: 'Answers dependent questions about SEC filings within ',
          emphasis: human !== null ? formatPointsDelta(human - sdk, 1).replace('+', '') : null,
          after: ' of the human-expert figure — and shows every step.',
        };

  const lede = isDemo
    ? 'One Claude Agent SDK session per conversation, six calculator tools, a prompt distilled from four optimised agents. This deployment holds no API key: every turn replays the real stage events, tool calls and timings captured in development.'
    : 'One Claude Agent SDK session per conversation, six calculator tools, a prompt distilled from four optimised agents. Every turn streams its stages live, with the dataset’s gold answer beside its own.';

  const proof: ProofStat[] = [
    {
      key: 'runtime',
      value: measured ? formatPercent(sdk) : NO_VALUE,
      label: measured
        ? `${questions ?? '—'} unseen questions · gate split`
        : 'the single-session arm has not been scored on the gate split',
      tone: measured ? 'text' : 'faint',
    },
    {
      key: 'human',
      value: human !== null ? formatPercent(human) : NO_VALUE,
      label: `human expert · paper, ${PAPER_HUMAN.evaluatedOn || 'published sample'}`,
      tone: 'violet',
    },
    {
      key: 'delta',
      value: verdict.deltaPp !== null ? formatPointsDelta(verdict.deltaPp / 100) : NO_VALUE,
      label:
        verdict.deltaPp !== null
          ? `over the optimised pipeline${verdict.baselineVersion ? ` (${verdict.baselineVersion})` : ''}, paired · p=${
              verdict.pValue !== null ? verdict.pValue.toExponential(0) : '—'
            }`
          : 'no cross-runtime gate has been run',
      tone:
        verdict.deltaPp === null ? 'faint' : verdict.deltaPp >= 0 ? 'good' : 'bad',
    },
  ];

  const caveat = measured
    ? 'A different question set from the paper, contamination not excluded, and model and architecture changed together — the caveats sit beside the number on Runtimes.'
    : null;

  const holdout =
    questions !== null && reports !== null
      ? `Every figure here is the sealed gate split: ${questions} questions across ${reports} conversations, fixed for the campaign. The holdout has never been opened, so nothing on this page is out-of-sample.`
      : 'Every figure here is the fixed gate split. The holdout has never been opened, so nothing on this page is out-of-sample.';

  return { headline, lede, proof, caveat, holdout, measured };
}

/** The judge card's one sentence, branching on significance the same way the admin banner does. */
export function judgeSentence(campaigns: CampaignsResponse | undefined): {
  headline: string;
  body: string;
  significant: boolean;
} | null {
  const v = campaigns?.judge?.verdict ?? null;
  if (!v) return null;
  const hb = formatPercent(v.high_band_accuracy, 2);
  const base = formatPercent(v.baseline_accuracy, 2);
  return {
    significant: Boolean(v.significant),
    headline: v.significant
      ? `${formatPointsDelta(v.delta_pp / 100)} — the band separates from it`
      : 'no effect',
    body: v.significant
      ? `High band ${hb} vs ${base} releasing everything — the interval clears the baseline, so the band separates from it. It withholds ${v.n_withheld} answers to remove ${v.n_failures_caught} wrong ones; ${v.n_false_alarms} of them were right.`
      : `High band ${hb} vs ${base} releasing everything — the interval contains the baseline, so the band fails to separate from it. It withholds ${v.n_withheld} answers to remove ${v.n_failures_caught} wrong ones; ${v.n_false_alarms} of them were right. Shipped as a caution, not a gate.`,
  };
}
