import { PageHead } from '@/kit/ui';

/** Slot 8 · what we know by hand. The s16 C4 page reads the settled verdicts; this is its frame. */
export default function Gold() {
  return (
    <div data-testid="gold-route">
      <PageHead
        eyebrow="gold review · what we know by hand"
        title={
          <>
            Gold labels the loop could not <em>trust</em>, and how each was settled
          </>
        }
        lead="Every diagnosis the teacher filed as gold_suspect, every case attribution left ambiguous, and the binary adjudication that settled it."
      />
    </div>
  );
}
