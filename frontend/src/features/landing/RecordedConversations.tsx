import { Link } from 'react-router-dom';
import { formatCount } from './format';
import type { RecordedConversation } from './useBoardData';

function CardSkeleton() {
  return (
    <div className="animate-pulse rounded-md border border-line bg-panel p-3">
      <div className="mb-2 h-2.5 w-24 rounded bg-panel-2" />
      <div className="mb-2 h-3 w-full rounded bg-panel-2" />
      <div className="h-2.5 w-20 rounded bg-panel-2" />
    </div>
  );
}

interface Props {
  conversations: RecordedConversation[] | undefined;
  loading: boolean;
  isDemo: boolean;
}

/**
 * Three doors into the product.
 *
 * A card is a whole conversation, not a question: the filing it is anchored
 * to, the question the conversation opens with, and how many turns follow.
 * Clicking one lands in `/chat` on that report, which is the shortest honest
 * path from "what is this" to "watch it work" — and the reason the landing
 * page is worth having at `/` at all.
 *
 * The "N ✓" half of the meta line is the recorded run's own score, and it is
 * omitted rather than assumed when the card came from the dataset list instead
 * of the pack.
 */
export function RecordedConversations({ conversations, loading, isDemo }: Props) {
  if (loading) {
    return (
      <div className="grid gap-2">
        <CardSkeleton />
        <CardSkeleton />
        <CardSkeleton />
      </div>
    );
  }

  if (!conversations || conversations.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-line-2 bg-panel p-3">
        <p className="type-small text-muted">
          No conversations available from this deployment. Open the chat and pick any filing from
          the report picker.
        </p>
      </div>
    );
  }

  return (
    <div className="grid gap-2">
      {conversations.map((c) => (
        <Link
          key={c.reportId}
          to={`/chat/${c.reportId}`}
          data-testid="recorded-conversation"
          data-rid={c.reportId}
          className="group block rounded-md border border-line bg-panel p-3 transition-colors hover:border-amber-line hover:bg-panel-2 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-amber"
        >
          {/*
            Deliberately not `.mono-caps`: that class uppercases, and a filing
            reference reads as `p.55`, not `P.55`. Same mono micro-size, same
            tracking, without the transform.
          */}
          <div className="type-num text-[10px] tracking-[0.08em] text-faint group-hover:text-amber">
            {c.label}
          </div>
          <div className="type-body mt-1 text-text">{c.firstQuestion || 'Open this filing'}</div>
          <div className="type-meta mt-1.5 text-faint">
            <span className="type-num">{formatCount(c.nTurns)}</span> turns
            {c.nCorrect !== null && (
              <>
                {' · '}
                <span className="type-num text-good">{formatCount(c.nCorrect)} ✓</span>
                {isDemo ? ' recorded' : ' in the recorded run'}
              </>
            )}
          </div>
        </Link>
      ))}
    </div>
  );
}
