import { PageHead } from '@/kit/ui';
import { BundleSection, LlmSection, PipelineSection } from '@/features/system/SectionArchitecture';
import {
  DeploySection,
  NewVersionSection,
  ObservabilitySection,
  OpenWorkSection,
} from '@/features/system/SectionOperations';
import { useSystemData } from '@/features/system/useSystemData';

/** Slot 7 · what the agent is. Assembled from the System page's sections (s16 C1); C4 rebuilds it. */
export default function Agent() {
  const data = useSystemData();
  return (
    <div data-testid="system-route">
      <PageHead
        eyebrow="agent · how it was built and how to operate it"
        title={
          <>
            Four agents, one choke point, one <em>contract</em>
          </>
        }
        lead="The pipeline and the single string that routes it, the one place a model is built, the fingerprint that makes a version mean something, what is observed, how it ships, and what is still broken."
      />
      <PipelineSection />
      <LlmSection data={data} />
      <BundleSection data={data} />
      <ObservabilitySection data={data} />
      <DeploySection data={data} />
      <NewVersionSection />
      <OpenWorkSection data={data} />
    </div>
  );
}
