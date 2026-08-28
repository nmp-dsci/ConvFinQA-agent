import { create } from 'zustand';
import * as api from './api';
import type { Health } from './types';

/**
 * Deployment mode, read once from `/healthz`.
 *
 * The same bundle ships to dev and to the public demo; this store is how the UI
 * finds out which one it is running in. Keeping it in one place means components
 * ask `isDemo` rather than each inventing its own detection, and there is never
 * a demo-specific build to drift from the real one.
 */
interface ModeState {
  health: Health | null;
  loading: boolean;
  error: string | null;
  ownerToken: string;
  load: () => Promise<void>;
  setOwnerToken: (token: string) => void;
}

export const useMode = create<ModeState>((set) => ({
  health: null,
  loading: false,
  error: null,
  ownerToken: api.getOwnerToken(),

  async load() {
    set({ loading: true, error: null });
    try {
      set({ health: await api.getHealth(), loading: false });
    } catch (err) {
      set({ error: String(err), loading: false });
    }
  },

  setOwnerToken(token: string) {
    api.setOwnerToken(token);
    set({ ownerToken: token });
  },
}));

/** True when this deployment refuses model calls and replays recordings. */
export function useIsDemo(): boolean {
  return useMode((s) => s.health?.mode === 'demo');
}
