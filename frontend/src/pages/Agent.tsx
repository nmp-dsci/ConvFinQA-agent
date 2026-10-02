import { Link } from 'react-router-dom';
import { PageHead } from '@/kit/ui';
import { PipelineDiagram } from '@/features/system/diagrams';
import { BundleSection, LlmSection, PipelineSection } from '@/features/system/SectionArchitecture';
import {
  DeploySection,
  NewVersionSection,
  ObservabilitySection,
  OpenWorkSection,
} from '@/features/system/SectionOperations';
import { useSystemData } from '@/features/system/useSystemData';
import '@/features/system/system.css';

/**
 * Slot 7 · what the agent is. The pipeline figure leads; the sections under it are the mechanism
 * (routing, the one place a model is built, the version fingerprint) and then the operation
 * (what is observed, how it ships, how a version is made, what is still broken).
 */
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
      >
        <p className="sys-action">
          <Link to="/agent/chat" className="btn">
            Ask a filing a chain of questions →
          </Link>
        </p>
      </PageHead>
      <PipelineDiagram />
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
