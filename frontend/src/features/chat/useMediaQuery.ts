import { useEffect, useState } from 'react';

/**
 * Read a media query in React.
 *
 * Used for the one thing CSS cannot express here: the inspector's drag-resized
 * width is an inline style, and an inline width must not survive into the
 * narrow layout where the inspector becomes a full-width drawer.
 */
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return false;
    return window.matchMedia(query).matches;
  });

  useEffect(() => {
    if (typeof window === 'undefined' || !window.matchMedia) return;
    const mql = window.matchMedia(query);
    const onChange = () => setMatches(mql.matches);
    onChange();
    mql.addEventListener('change', onChange);
    return () => mql.removeEventListener('change', onChange);
  }, [query]);

  return matches;
}
