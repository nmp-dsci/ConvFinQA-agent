import { PageHead } from '@/kit/ui';
import { versionLabel } from '@/features/admin/lib';
import { formatPercent } from '@/features/landing/format';
import {
  BenchmarkSection,
  LeaderboardSection,
  PaperSection,
} from '@/features/system/SectionPaper';
import { BASELINES } from '@/features/system/paper';
import { useSystemData } from '@/features/system/useSystemData';
import type { SystemData } from '@/features/system/useSystemData';
import '@/features/system/system.css';

const HUMAN = BASELINES.find((row) => row.ceiling);
const HUMAN_EXE = HUMAN?.exe ? Number.parseFloat(HUMAN.exe) / 100 : null;

/**
 * The headline, derived: our best measured runtime against the paper's human expert. It is only
 * written as "past" when the figure is, and it always names the version it is about.
 */
function headline(data: SystemData) {
  const arms = [data.runtimes?.agent_sdk, data.runtimes?.pipeline].filter(
    (arm): arm is NonNullable<typeof arm> => typeof arm?.accuracy === 'number',
  );
  const best = arms.sort((a, b) => (b.accuracy ?? 0) - (a.accuracy ?? 0))[0];
  if (!best || best.accuracy === null || HUMAN_EXE === null) {
    return {
      title: (
        <>
          The paper&rsquo;s systems and ours on one axis, against its <em>human</em> expert
        </>
      ),
      lead: 'Chen et al. (2022) report execution accuracy for fine-tuned FinQANet, prompted GPT-3 and two human panels. Our runtimes are not scored on this deployment yet, so their rows say why instead of showing a number.',
    };
  }
  const who = versionLabel(best.version) || 'our best runtime';
  const acc = formatPercent(best.accuracy);
  const human = formatPercent(HUMAN_EXE, 2);
  const past = best.accuracy >= HUMAN_EXE;
  return {
    title: past ? (
      <>
        {who} scores {acc}, past the paper&rsquo;s <em>human</em> expert at {human}
      </>
    ) : (
      <>
        {who} scores {acc}, short of the paper&rsquo;s <em>human</em> expert at {human}
      </>
    ),
    lead: `Execution accuracy on ${data.gateQuestions ?? 'the'} unseen gate-split questions, beside every system Chen et al. (2022) report. Different questions and a different era of model: the caveat under the figure is part of the result.`,
  };
}

/** Slot 4 · who else has tried: one axis, the paper's systems and ours, the caveat under it. */
export default function Leaderboard() {
  const data = useSystemData();
  const head = headline(data);
  return (
    <div data-testid="leaderboard-route">
      <PageHead eyebrow="leaderboard · the paper's systems and ours" title={head.title} lead={head.lead} />
      <LeaderboardSection data={data} />
      <BenchmarkSection data={data} />
      <PaperSection />
    </div>
  );
}
