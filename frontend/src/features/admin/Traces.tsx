import { Placeholder } from './Placeholder';

export default function Traces() {
  return (
    <Placeholder title="Traces" phase={4} testId="admin-traces">
      Every turn the system has answered, stage by stage: what each agent saw, what it returned,
      which calculator tools ran, and how long it took. Filterable by report and by source, so
      serving turns and demo replays stay separable.
    </Placeholder>
  );
}
