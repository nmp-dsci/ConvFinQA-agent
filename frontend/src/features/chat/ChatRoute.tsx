import { useEffect } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { ChatPanel } from '../../components/ChatPanel';
import { ReportPicker } from '../../components/ReportPicker';
import { Sidebar } from '../../components/Sidebar';
import { useStore } from '../../store';
import { ChatHeader } from './ChatHeader';

/**
 * The chat surface, unchanged in behaviour from the pre-redesign console —
 * Phase 3 rebuilds its interior. Phase 0 only moves it under the router and
 * gives it a URL.
 *
 * `/chat/:reportId` selects that conversation, which is what makes a chat
 * linkable at all: before this, "the conversation about VLO 2011 page 126"
 * could only be reached by clicking through the picker.
 */
export function ChatRoute() {
  const params = useParams();
  const navigate = useNavigate();
  const activeReportId = useStore((s) => s.activeReportId);
  const selectReport = useStore((s) => s.selectReport);

  const routeReportId = params['*'] ? decodeURIComponent(params['*']) : '';

  useEffect(() => {
    if (!routeReportId || routeReportId === activeReportId) return;
    void selectReport(routeReportId);
  }, [routeReportId, activeReportId, selectReport]);

  // Keep the URL honest when the user switches conversations from the sidebar
  // or the picker, so a copied link points at what is actually on screen.
  useEffect(() => {
    if (!routeReportId) return;
    if (activeReportId && activeReportId !== routeReportId) {
      navigate(`/chat/${activeReportId}`, { replace: true });
    }
  }, [activeReportId, routeReportId, navigate]);

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden">
      <ChatHeader />
      <div className="grid min-h-0 flex-1 grid-cols-[320px_1fr] overflow-hidden">
        <Sidebar />
        <div className="flex min-h-0 min-w-0 flex-col overflow-x-hidden">
          <ChatPanel />
        </div>
      </div>
      <ReportPicker />
    </div>
  );
}
