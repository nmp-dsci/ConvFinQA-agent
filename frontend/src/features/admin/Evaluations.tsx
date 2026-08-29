import { Placeholder } from './Placeholder';

export default function Evaluations() {
  return (
    <Placeholder title="Evaluations" phase={4} testId="admin-evaluations">
      Accuracy per version sliced by turn type, conversation type and question order, with the
      split membership visible and every number drillable down to the questions behind it. Reads
      the committed prediction CSVs, so it works with no key and no tracking server.
    </Placeholder>
  );
}
