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
