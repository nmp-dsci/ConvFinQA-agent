import { useEffect, useState } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { DataPanel } from './components/DataPanel';
import { EvalPanel } from './components/EvalPanel';
import { ExperimentsPanel } from './components/ExperimentsPanel';
import { LandingPage } from './components/LandingPage';
import { ReportPicker } from './components/ReportPicker';
import { ResearchPanel } from './components/ResearchPanel';
import { Sidebar } from './components/Sidebar';
import { TopBar, type AppTab } from './components/TopBar';
import { TracesPanel } from './components/TracesPanel';
import { useMode } from './modeStore';
import { useStore } from './store';

const ENTERED_KEY = 'convfinqa.entered';

export default function App() {
  const loadReports = useStore((s) => s.loadReports);
  const loadMode = useMode((s) => s.load);
  const [activeTab, setActiveTab] = useState<AppTab>('chat');

  // The landing page is the front door, but only the first time — bouncing a
  // returning user back through it on every reload is friction, not welcome.
  const [entered, setEntered] = useState(() => {
    try {
      return window.sessionStorage.getItem(ENTERED_KEY) === '1';
    } catch {
      return false;
    }
  });

  useEffect(() => {
    void loadMode();
    void loadReports();
  }, [loadMode, loadReports]);

  function enter() {
    try {
      window.sessionStorage.setItem(ENTERED_KEY, '1');
    } catch {
      // Private browsing — the landing page simply shows again next reload.
    }
    setEntered(true);
  }

  if (!entered) {
    return (
      <div className="h-full bg-bg text-textMain overflow-hidden">
        <LandingPage onEnter={enter} />
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col bg-bg text-textMain overflow-hidden">
      <TopBar activeTab={activeTab} onTabChange={setActiveTab} onHome={() => setEntered(false)} />
      <div className="flex-1 min-h-0 overflow-hidden">
        {activeTab === 'chat' && (
          <div className="h-full grid grid-cols-[320px_1fr] min-h-0 overflow-hidden">
            <Sidebar />
            <div className="flex flex-col min-h-0 min-w-0 overflow-x-hidden">
              <ChatPanel />
            </div>
          </div>
        )}
        {activeTab === 'data' && <DataPanel />}
        {activeTab === 'traces' && <TracesPanel />}
        {activeTab === 'experiments' && <ExperimentsPanel />}
        {activeTab === 'research' && <ResearchPanel />}
        {activeTab === 'eval' && <EvalPanel />}
      </div>
      <ReportPicker />
    </div>
  );
}
