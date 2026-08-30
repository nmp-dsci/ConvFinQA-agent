// Self-hosted, no CDN: the demo container must render identically with no
// outbound network at all. Upright weights only — nothing in this UI is
// italic, and the mono faces are pinned to the latin subset because every
// string it renders is an id, a number or a program.
import '@fontsource-variable/ibm-plex-sans/wght.css';
import '@fontsource/ibm-plex-mono/latin-400.css';
import '@fontsource/ibm-plex-mono/latin-500.css';
import '@fontsource/ibm-plex-mono/latin-600.css';

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
