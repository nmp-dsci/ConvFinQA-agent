// Self-hosted, no CDN: the demo container must render identically with no
// outbound network at all. The house's three voices (ui/DESIGN.md): Serif for the
// h1 and the lead (with the one italic <em>), Sans for the interface, Mono for data.
// Latin subset only: every string this UI renders is English, an id or a number.
import '@fontsource/ibm-plex-serif/latin-400.css';
import '@fontsource/ibm-plex-serif/latin-400-italic.css';
import '@fontsource/ibm-plex-serif/latin-500.css';
import '@fontsource/ibm-plex-sans/latin-400.css';
import '@fontsource/ibm-plex-sans/latin-500.css';
import '@fontsource/ibm-plex-sans/latin-600.css';
import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';

import { QueryClientProvider } from '@tanstack/react-query';
import React from 'react';
import ReactDOM from 'react-dom/client';
import { RouterProvider } from 'react-router-dom';
import { installPreloadErrorGuard } from './app/preloadGuard';
import { router } from './app/router';
import { queryClient } from './lib/queryClient';
import './styles.css';

installPreloadErrorGuard();

const root = document.getElementById('root');
if (!root) throw new Error('Root element #root not found');

ReactDOM.createRoot(root).render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </React.StrictMode>
);
