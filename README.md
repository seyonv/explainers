# Explainers

**[seyonv.github.io/explainers](https://seyonv.github.io/explainers/)**

My visual notes cards on technical concepts, all on one page. Each card has a worked example with
real, computed numbers. Courses open as a gallery you can page through with ← →.

Everything here is made with two Claude Code skills:
[concept-explainer](https://github.com/seyonv/concept-explainer) makes one card, and
[concept-curriculum](https://github.com/seyonv/concept-curriculum) makes a whole course.

## How it's organised

```
explainers/
  index.html            the hub page (generated, don't edit)
  llm-latency/          a course: has its own index.html gallery
  nucleus-sampling/     a single card
  …
```

- A folder with an `index.html` is a **course**; its tile shows the card count and opens its gallery.
- A folder with one card is a **single card**.
- The title and description come from each folder's page. To override them, add
  `<folder>/explainer.json` with `{"title": "…", "description": "…", "date": "YYYY-MM-DD"}`.

## Adding an explainer

Put the folder here, then:

```bash
./publish.sh <folder>
```

This rebuilds `index.html` from the folders (`build-hub.mjs` + `hub-template.html`), commits and
pushes. GitHub Pages updates in about a minute. The skills do this automatically when a card or
course is finished.

`.nojekyll` must stay: without it, GitHub Pages drops every file whose name starts with `_`, such
as `_overview.html`.

## Course navigation

`course-nav.mjs` gives every course the same navigation. A course is any folder whose `index.html`
has a `GROUPS` list. Each card gets breadcrumbs with prev/next at the top and a big "next up"
preview at the bottom, and each course index gets a contents sidebar in the reader. `publish.sh`
runs it on every publish (papers included), so a new course picks it up with no extra step.

The `perf-*` folders form one linked series. `perf-series.json` lists the courses in order, what each course needs, and each card's prerequisites. For those courses the navigation also adds a course strip, a "before this" line on each card, and "next up" links that cross into the next course. Adding a course to the series only needs an entry in `perf-series.json`. `cs-series.json` does the same for the `cs-*` courses ("The core CS canon").

On the hub, each series shows as one wide tile (on its map folder) that lists its courses. Its member courses are hidden from the grid unless a search matches them.

## Study Mode

Cards teach through a sequence of study stages: Prime → Learn → Encode → Compress → Retrieve → Compare → Predict → Apply → Reflect. [learning-principles/](learning-principles/learning-principles.html) explains why.

- Each card's stages live in `<course>/_study/<slug>.json`. `study-lib.mjs` has the schema and the rules.
- `study-build.mjs` injects a **Prime** block under the card's subtitle and a CSS-only **Study this card** stepper before the footer (cards can't run script). Each stage appears only once the previous one is opened.
- `publish.sh` runs it before `course-nav.mjs`. Use `--card course/slug` to build a single card.
- `verify-study.mjs <course>` checks the JSON, flagging generic or repeated questions, missing causal chains and links to cards that don't exist.
- For interactive tutoring, `/study <course>/<slug>` in Claude Code walks through the stages one at a time and grades your answers.

## Papers

Cards made from a paper (via `concept-curriculum` pointed at an arXiv paper) land in a
`paper-<id>/` folder, each with an `explainer.json` that has `"kind": "paper"`. `sync-papers.mjs`
keeps these folders in sync with the paper source list; it only touches `paper-*` folders and
leaves everything else alone. `publish.sh` runs `node sync-papers.mjs` before `build-hub.mjs`, so
every publish re-syncs papers first. A launchd job (`launchd/com.seyonv.explainers-sync.plist`,
installed separately) runs `./publish.sh papers` every 30 minutes so new papers show up without
manual intervention.
