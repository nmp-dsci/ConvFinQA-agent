import { PageHead } from '@/kit/ui';
import { BenchmarkSection, PaperSection } from '@/features/system/SectionPaper';
import { useSystemData } from '@/features/system/useSystemData';

/** Slot 4 · who else has tried. Assembled from the paper sections (s16 C1); C3 rebuilds it. */
export default function Leaderboard() {
  const data = useSystemData();
  return (
    <div data-testid="leaderboard-route">
      <PageHead
        eyebrow="leaderboard · the paper's baselines and ours"
        title={
          <>
            Where this agent lands among the ConvFinQA paper&rsquo;s systems and its <em>human</em> expert
          </>
        }
        lead="Chen et al. (2022) report execution accuracy for fine-tuned FinQANet, GPT-3 few-shot and two human experts; our runtimes are scored on the same metric, with the caveats that make the comparison fair."
      />
      <PaperSection />
      <BenchmarkSection data={data} />
    </div>
  );
}
