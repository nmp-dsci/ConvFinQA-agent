import { Link } from 'react-router-dom';
import { Empty, Loading, Rate } from '@/kit/ui';
import { formatCount } from './format';
import type { RecordedConversation } from './useBoardData';

interface Props {
  conversations: RecordedConversation[] | undefined;
  loading: boolean;
  isDemo: boolean;
}

/**
 * Doors into the product, as a house table.
 *
 * A row is a whole conversation, not a question: the question the conversation opens with (the
 * link), the filing it is anchored to under it, how many turns follow, and how the recorded run did.
 * Clicking one lands in `/agent/chat` on that report, which is the shortest honest path from "what
 * is this" to "watch it work" — and the reason the Overview is worth having at `/` at all.
 *
 * The score column is the recorded run's own, and it says so rather than being assumed when the
 * row came from the dataset list instead of the pack.
 */
export function RecordedConversations({ conversations, loading, isDemo }: Props) {
  if (loading) return <Loading what="recorded conversations" />;

  if (!conversations || conversations.length === 0) {
    return (
      <Empty>
        No conversations available from this deployment. Open the chat and pick any filing from the
        report picker.
      </Empty>
    );
  }

  return (
    <div className="tw">
      <table>
        <caption>
          {isDemo
            ? 'Recorded in development and replayed here without a model call'
            : 'Opened live on this deployment; the score is the recorded run’s'}
        </caption>
        <thead>
          <tr>
            <th>Opening question · filing</th>
            <th className="num">Turns</th>
            <th>{isDemo ? 'Right when recorded' : 'Right in the recorded run'}</th>
          </tr>
        </thead>
        <tbody>
          {conversations.map((c) => (
            <tr key={c.reportId}>
              <td className="q">
                <Link
                  to={`/agent/chat/${c.reportId}`}
                  data-testid="recorded-conversation"
                  data-rid={c.reportId}
                >
                  {c.firstQuestion || 'Open this filing'}
                </Link>
                {/* a filing reference reads as `p.55`, not `P.55`, so not `.label` */}
                <span className="path">{c.label}</span>
              </td>
              <td className="num">{formatCount(c.nTurns)}</td>
              <td>
                {c.nCorrect !== null ? (
                  <Rate passed={c.nCorrect} n={c.nTurns} unit="turns" />
                ) : (
                  <span className="muted">— listed from the dataset, never recorded</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
