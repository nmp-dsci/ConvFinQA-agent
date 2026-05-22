import { useEffect, useState } from 'react';
import { ChatPanel } from './components/ChatPanel';
import { EvalPanel } from './components/EvalPanel';
import { ReportPicker } from './components/ReportPicker';
import { Sidebar } from './components/Sidebar';
import { TopBar, type AppTab } from './components/TopBar';
import { useStore } from './store';

export default function App() {
  const loadReports = useStore((s) => s.loadReports);
  const [activeTab, setActiveTab] = useState<AppTab>('chat');

  useEffect(() => {
    void loadReports();
  }, [loadReports]);

  return (
    <div className="h-full flex flex-col bg-bg text-textMain overflow-hidden">
      <TopBar activeTab={activeTab} onTabChange={setActiveTab} />
      {activeTab === 'chat' ? (
        <div className="flex-1 grid grid-cols-[320px_1fr] min-h-0 overflow-hidden">
          <Sidebar />
          <div className="flex flex-col min-h-0 min-w-0 overflow-x-hidden">
            <ChatPanel />
          </div>
        </div>
      ) : (
        <EvalPanel />
      )}
      <ReportPicker />
    </div>
  );
}
