import { useNavigate } from 'react-router-dom';
import { LandingPage } from '../../components/LandingPage';

/**
 * The public front door, still the pre-redesign `LandingPage` — Phase 2 owns
 * rebuilding it. Phase 0 only replaces its sessionStorage "have I entered yet"
 * gate with a route: the CTA navigates instead of flipping a boolean, which is
 * what lets a link point at either the story or the console.
 */
export function LandingRoute() {
  const navigate = useNavigate();
  return (
    <div className="h-full overflow-y-auto bg-panel">
      <LandingPage onEnter={() => navigate('/chat')} />
    </div>
  );
}
