import { useParams } from 'react-router-dom';
import { Placeholder } from './Placeholder';

export default function TraceDetail() {
  const { traceId } = useParams();
  return (
    <Placeholder title="Trace detail" phase={4} testId="admin-trace-detail">
      The four-stage inspector for a single turn
      {traceId ? (
        <>
          {' '}
          (<span className="font-mono text-xs text-text">{traceId}</span>)
        </>
      ) : null}
      : stage inputs and outputs, the tool trajectory, token counts and latency, and the bundle
      that produced it.
    </Placeholder>
  );
}
