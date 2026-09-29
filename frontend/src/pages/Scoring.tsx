import { PageHead } from '@/kit/ui';
import { EvaluationSection, PromotionSection } from '@/features/system/SectionContract';
import { useSystemData } from '@/features/system/useSystemData';

/** Slot 3 · how a turn is judged. Assembled from the contract sections (s16 C1); C3 rebuilds it. */
export default function Scoring() {
  const data = useSystemData();
  return (
    <div data-testid="scoring-route">
      <PageHead
        eyebrow="scoring · how a turn is judged"
        title={
          <>
            A turn counts when its executed answer matches gold, and a challenger moves only on{' '}
            <em>significant</em> unseen evidence
          </>
        }
        lead="Execution accuracy per turn, the two-number rule for reporting it, and the one-sided cluster-corrected McNemar gate every promotion has to pass."
      />
      <EvaluationSection data={data} />
      <PromotionSection data={data} />
    </div>
  );
}
