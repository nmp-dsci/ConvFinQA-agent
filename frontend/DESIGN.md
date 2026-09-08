# DESIGN.md — the Console's design brief

Read before touching anything under `src/`. The plan this brief comes from is
`.lavish/s14_design-upgrade-plan.html`; the rubric it is scored on is the one the
portfolio site uses (twelve dimensions, weighted, max 145). The app scored 82 on
8 Sep 2026. The ship bar is 130.

## Who this is for

1. **A technical reviewer with 90 seconds.** Arrives from the portfolio's R1–R9
   matrix or a CV link. Wants: what was built, how it was measured, whether the
   numbers are honest, and proof it runs. Their objection is "a chatbot with a
   dashboard is not production engineering". The answer is the loop, the gate
   and the readiness scorecard — shown, not described.
2. **The operator, in dev.** Runs live turns on the Agent SDK, reads traces,
   promotes or refuses versions. Needs density and keyboard reach. The
   instrument-panel identity exists for this person and survives every redesign.
3. **The public demo.** No key; replays eight recorded conversations; every
   write answers 403. A visitor must understand within one screen that the chat
   is a faithful replay and the admin pages are live against committed evidence.

## The one thing they must leave knowing

> The loop took a four-agent pipeline from 77.1% to 81.7% on a sealed split;
> one Claude session reached 90.5% on the same split — above the paper's 89.4%
> human figure, with the caveats attached — and the confidence judge tried
> afterwards made no measurable difference.

Six beats, in order, each with its proof page: architecture → the loop → evals →
human level with the caveat → working Q/A (dev) and replay (prod) → the judge,
tried last. The nav is this order. Every page states its own beat in one
sentence in its header, generated from the same data it renders.

## The never-do list

- No rendered text under 11px; no running text under 13px. Dense tables floor at 12.
- No `text-[Npx]`. Sizes come from the eight `.type-*` steps in `tokens.css`.
- No hex colour outside `tokens.css`. No `text-accent` / `bg-accent` / legacy aliases.
- No icon-only navigation. Every nav item has a visible label at ≥ 1024px and a
  reachable one below.
- No colour-only status. Lamps and bands carry a shape or a word as well.
- No unlabelled em dash. An absent value prints `—` **and** the reason.
- No ghost primary. One solid amber CTA per screen; secondaries are quiet.
- No page without a main character, and no page that ends without a next step.
- No number written in a component. Every figure is read from a query or a
  committed file; every verdict sentence is a pure function with a test per branch.
- No claim of human-level reasoning. The 90.5% carries "different question set,
  contamination not excluded, model and architecture changed together" wherever
  it is drawn next to the human figure.
- No motion that does not honour `prefers-reduced-motion`; no transition over 300ms.

## Tokens

Colour: unchanged — both themes from the same names, measured ≥ 4.5:1 (faint
4.65:1 dark / 4.72:1 light). Elevation is monotonic ground → panel → panel-2.

Type (px, absolute on purpose — an instrument panel does not reflow with the
reader's root size):

| step | px | use |
| --- | --- | --- |
| `--fs-micro` | 11 | `.mono-caps` labels |
| `--fs-meta` | 12 | table meta, captions, chips |
| `--fs-small` | 13 | dense secondary prose |
| `--fs-body` | 14 | default UI reading size |
| `--fs-read` | 16 | reading prose (Architecture), 68ch, lh 1.6 |
| `--fs-lede` | 17 | the one sentence under a headline |
| `--fs-h2` | 24 | page and section headings |
| `--fs-hud` | 30 | the number on a tile |
| `--fs-display` | 28–40 fluid | the H1 on the landing and Architecture |

Motion: `--dur` 220ms, `--ease` ease-out; zero under reduced motion. Focus:
2px amber ring with 2px offset on every focusable, global.

## Enforcement

`npm run lint:design` (`scripts/design_lint.mjs`) fails the build on hex
colours and legacy aliases anywhere in `src/`, and ratchets raw `text-[Npx]`
counts per file against `design-baseline.json` — a file may only go down. Run
`npm run lint:design -- --update` after removing sizes to lower the baseline.
`scripts/design_probe.mjs` (P6) measures rendered pages: text styles per route,
smallest rendered size, overflow at 390/768/1280/1600.
