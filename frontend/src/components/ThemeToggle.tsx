import { Moon, Sun } from 'lucide-react';
import { Tooltip, TooltipContent, TooltipTrigger } from '@/components/ui/tooltip';
import { useTheme } from '@/lib/theme';

export function ThemeToggle() {
  const { theme, toggle } = useTheme();
  const next = theme === 'dark' ? 'light' : 'dark';

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          type="button"
          onClick={toggle}
          data-testid="theme-toggle"
          data-theme-state={theme}
          aria-label={`Switch to ${next} theme`}
          className="flex size-7 items-center justify-center rounded-md text-faint transition-colors hover:bg-panel-2 hover:text-text"
        >
          {theme === 'dark' ? (
            <Moon className="size-4" aria-hidden />
          ) : (
            <Sun className="size-4" aria-hidden />
          )}
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom">Switch to {next} theme</TooltipContent>
    </Tooltip>
  );
}
