import { FileText, Plus, Search } from 'lucide-react';
import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CommandDialog,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
  CommandSeparator,
  CommandShortcut,
} from '@/components/ui/command';
import { useStore } from '../../store';
import { NAV_ITEMS } from '../../app/nav';
import { shortRid } from './format';

/** Every page but the chat itself, in story order — the same list the header reads. */
const PAGES = NAV_ITEMS.filter((item) => item.to !== '/agent/chat');

/**
 * ⌘K. Four things a person actually wants mid-conversation: change filing,
 * start a new one, jump to a page. (The theme follows the OS, as every house app does.)
 *
 * Filings are listed directly rather than behind a submenu — recall beats
 * browsing once you know the ticker, and the picker dialog is still there for
 * when you do not.
 */
export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const navigate = useNavigate();
  const reports = useStore((s) => s.reports);
  const selectReport = useStore((s) => s.selectReport);
  const openPicker = useStore((s) => s.openPicker);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key.toLowerCase() !== 'k') return;
      if (!event.metaKey && !event.ctrlKey) return;
      event.preventDefault();
      setOpen((prev) => !prev);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  const run = (action: () => void) => {
    setOpen(false);
    action();
  };

  return (
    <CommandDialog
      open={open}
      onOpenChange={setOpen}
      title="Command palette"
      description="Switch filing, start a conversation, or jump to an instrument page"
    >
      <CommandInput placeholder="Type a filing, a page, or a command…" />
      <CommandList>
        <CommandEmpty>Nothing matches.</CommandEmpty>

        <CommandGroup heading="Conversation">
          <CommandItem
            value="new conversation"
            onSelect={() => run(() => openPicker())}
            data-testid="palette-new-conversation"
          >
            <Plus aria-hidden />
            New conversation
          </CommandItem>
          <CommandItem value="browse all filings" onSelect={() => run(() => openPicker())}>
            <Search aria-hidden />
            Browse all filings…
          </CommandItem>
        </CommandGroup>

        <CommandSeparator />

        <CommandGroup heading="Pages">
          {PAGES.map((page) => (
            <CommandItem
              key={page.to}
              value={`${page.label} ${page.to}`}
              onSelect={() => run(() => navigate(page.to))}
            >
              {page.label}
              <CommandShortcut>{page.to}</CommandShortcut>
            </CommandItem>
          ))}
        </CommandGroup>


        {reports.length > 0 && (
          <>
            <CommandSeparator />
            <CommandGroup heading="Filings">
              {reports.slice(0, 200).map((rid) => (
                <CommandItem
                  key={rid}
                  value={rid}
                  onSelect={() => run(() => void selectReport(rid))}
                >
                  <FileText aria-hidden />
                  <span className="type-num type-small truncate">{rid}</span>
                  <CommandShortcut>{shortRid(rid)}</CommandShortcut>
                </CommandItem>
              ))}
            </CommandGroup>
          </>
        )}
      </CommandList>
    </CommandDialog>
  );
}
