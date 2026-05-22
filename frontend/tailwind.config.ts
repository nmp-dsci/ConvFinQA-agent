import type { Config } from 'tailwindcss';

const config: Config = {
  darkMode: 'class',
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      colors: {
        bg: '#0b141a',
        panel: '#111b21',
        panel2: '#202c33',
        accent: '#005c4b',
        accent2: '#00a884',
        bubbleUser: '#005c4b',
        bubbleAssistant: '#202c33',
        textMain: '#e9edef',
        textMuted: '#8696a0',
        danger: '#f15c6d',
      },
    },
  },
  plugins: [],
};

export default config;
