# DESIGN.md — the house brief ConvFinQA-agent is built to

The source of truth for how an nmp-ai-portfolio agent app looks and reads: its frontend,
its `.lavish/` review artifacts and its README figures. It is the portfolio site's brief
(`nmp-dsci.github.io/DESIGN.md`, direction "Field Guide") applied to an app, carved from
DataAgentBench's explorer (the reference implementation) and tau2-loop's viewer. If the
palette here and the site's `assets/css/tokens.css` disagree, the site wins — `make ui-check`
fails until this kit is fixed.

A project copies nothing by hand. `make ui-new PROJECT=<id>` scaffolds a frontend from
`ui/template/`; `make ui-sync PROJECT=<id>` brings its `src/kit/` up to the current kit;
`make ui-check` reports who is on which kit. The `platform-ui` skill drives all three.

---

## 1. What an app is for

**Reader:** the author iterating on an agent, and a reviewer scanning for 60–90 seconds. Both
scan headline to headline and read a quarter of the words.

**The one job:** make the result, the risk or the number obvious at a glance, with its evidence
directly under it, and let the reader find any failure and its trace in two clicks.

**Not:** a dashboard template, a pitch deck, or a generated-looking page.

## 2. The eight slots

Every app has the same eight tabs, in the same order, because the story is the same: an agent
optimised against a benchmark, with the work shown.

| # | Slot | Answers | DataAgentBench | tau2-loop | ConvFinQA |
|---|---|---|---|---|---|
| 1 | Overview | the result, with baseline and caveat | Overview | Overview | Overview |
| 2 | Data | what the benchmark is | Datasets & queries | Domains & tasks | Filings & questions |
| 3 | Scoring | how an answer is judged | Validators | Rubric | Scoring |
| 4 | Leaderboard | who else has tried, on one axis | Leaderboard | Leaderboard | Leaderboard |
| 5 | Runs | what we ran, and every trace | Runs | Runs | Runs |
| 6 | Optimise | how we improve: rounds, gates, promotions | Optimise | Optimise | Optimise |
| 7 | Agent | what the agent is, and running it | Agent | Agent | Agent |
| 8 | Review | what we know by hand | Golden | Review | Gold review |

Rename a slot to the benchmark's own noun; never reorder, merge or drop one. A page that does not
fit a slot is a section of one, reached by a lens (`?lens=`), not a ninth tab. Old addresses
redirect (routes as data, redirect loaders, `routes.test.tsx` lands every one of them).

## 3. The rules

### Presentation mode
Every page is a presentation. The unit is a slide: **assertion headline** (a claim, with its
number) → **visual** (the chart, diagram or table that is the evidence) → **one insight line**.

- Headings state claims, never topics: `3 · Labels — ten gold tasks are all anyone gets`.
  Banned: `Architecture`, `Cost`, `Overview`, `Setup`, `Results`, or any bare noun.
- Order: outcome (with baseline and caveat) → the visual → the mechanism → the setup.
- Every figure is introduced before it is read (`Figure`: a mono `fig-title`, the frame, a
  caption stating the takeaway first and the source path second).
- Every number carries its baseline or denominator in the same sentence (`Rate`: never a bare %).

### Type
- Three voices from IBM Plex: **Serif** for `h1` and leads, **Sans** for the interface, **Mono**
  for data (numbers, paths, keys, status words). Mono is never prose; serif is never a control.
- One decorated keyword per page, only inside the `h1`: a single `<em>`, italic, in `--accent`.
- One scale, `--t-2 … --t4`. Nothing smaller than `--t-2` (12.8px), including SVG labels.

### Colour
- 60/30/10: surfaces (`--bg --panel --band`), ink (four steps), accent.
- **`--accent` is the project's colour and means "act on this"**: links, the one solid button,
  the active tab, the brand's second word, the `<em>`. Never decoration, never status.
  Projects pick one from `ui/accents.yaml` (green DataAgentBench, red tau2-loop, blue ConvFinQA);
  a new one is added there with its contrast receipts before any project uses it.
- **`--ok` means passed, `--red` means failed, `--amber` means partial.** The same three in every
  app, whatever the accent — a pass never renders red because the brand is red.
- `--line-3` dashed means designed, not built.
- Status is never colour alone: a glyph and a word (`Status`: ● passed · ○ not run · ◐ partial · ✕ failed).
- Light is the reference palette; dark redefines colour only; the app follows the OS.
- No purple, no Inter, no gradient, no pastel panel.

### Space, shape, motion
- One 4/8 ladder (`--s1 … --s9`), one radius (`--r`), one small radius (`--r-sm`), one shadow.
- Cards sit on the page without borders; space separates things, not lines.
- Motion is micro only (300ms `--dur`); `prefers-reduced-motion` zeroes it. Focus is always a
  visible 2px `--focus` ring.

### Tables, figures
- Semantic `<table>` in a `.tw` scroll wrapper; mono uppercase `thead`; `td.num` tabular;
  `td.sub` for the row identity with a `.path` under it; `tr.pro` for the promoted row;
  `th.sortable` / `th.sorted` for sortable columns.
- Figures are hand-authored inline SVG (`.dia` vocabulary), coloured through tokens, sized by
  `viewBox`. Never draw a number that is not in a committed file.

## 4. Never do this

1. No ghost primary action: one solid `--accent` button per screen; the secondary is a link.
2. No text under `--t-2`. 3. No accent outside action, link, active and brand.
4. No placeholder for a result that does not exist. An absent value is `—` with its reason.
5. No headline metric without its baseline. 6. No paragraph over 80 words; no topic heading.
7. No icon-only control; no standalone control under 24×24px.
8. No side-scrolling page body; only tables, code and figures scroll, each in its own wrapper.
9. No hand-maintained duplicate of a list that exists in data.
10. No Tailwind, no component library styling: plain CSS on tokens (`design_lint.mjs` enforces it).

## 5. The kit boundary

| Where | Owner | What |
|---|---|---|
| `frontend/src/kit/` | the kit | `tokens.base.css`, `kit.css`, `Shell.tsx`, `ui.tsx`, `url.ts`, `http.ts`, `format.ts`, `harness.ts`, `design_lint.mjs`. Overwritten by `make ui-sync`; never edited in place (`ui-check` reports DRIFT). |
| `frontend/src/theme.css` | the project | the accent triple from `ui/accents.yaml` and any recorded extension token |
| `frontend/src/app.css` | the project | components the kit does not have |
| `frontend/src/styles.css` | the project | `@import` order: `kit/tokens.base.css` → `theme.css` → `kit/kit.css` → `app.css` |
| everything else | the project | routes, pages, API types, behaviour libraries |

A pattern that two projects need moves into the kit: change `ui/kit/`, bump `ui/VERSION`, then
`make ui-sync` each project on its own branch.

## 6. Checks

| Check | How |
|---|---|
| Kit version, drift, accent, palette parity with the site | `make ui-check` (all) or `make ui-check ARGS="--only P1"` |
| Hex outside tokens, text under 12.8px, banned fonts, Tailwind | `npm run lint:design` in the project (CI) |
| Every address lands | `routes.test.tsx` with `kit/harness.ts` |
| Contrast, both themes | `uv run --no-project python ../nmp-dsci.github.io/scripts/contrast_audit.py` for a new accent |
| Layout at 390 / 900 / 1280px, light and dark | screenshots before a UI PR merges |

---

## 7. ConvFinQA-agent: what is this project's own

The brief above is the kit's (`nmp-central-ai/ui/DESIGN.md`, kit 0.1.0) and wins on any
disagreement. This section records only what is ConvFinQA's.

- **Accent: blue.** `--accent #1F4E8C` / dark `#7EA8E4` (`frontend/src/theme.css`, from
  `ui/accents.yaml`). The Console's amber accent is gone; `--amber` now means *partial* only.
- **The slots, named for this benchmark:** Overview · Filings & questions · Scoring · Leaderboard ·
  Runs · Optimise · Agent · Gold review (`frontend/src/app/nav.ts`). Runs, Optimise and Agent
  carry a chip bar of sub-pages.
- **Chat is the Agent slot's working surface**, at `/agent/chat/*` — the one `wide` route
  (full-height, three panes and the Inspector, no footer).
- **Extensions in `frontend/src/app.css`** (candidates for the kit): lamps (a labelled status
  chip; dashed = replayed), panels with their source path, stat cells, provenance origins, the
  overlays (dialog, sheet, tooltip, ⌘K palette — Radix/cmdk underneath).
- **Kept from the Console, ConvFinQA-only:** the ⌘K palette inside chat; the Inspector's
  drag-resize; the `WriteGate` fieldset for demo mode.
- **Every absent value prints `—` and its reason**; every version prints through `versionLabel`
  (`llm-v8`, `sdk-v1`, `judge-j1`); no human-level claim without its caveat.
