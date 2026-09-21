// Pure helpers for turning an iterate_visuals_for_papers paper into a hub course folder.
export function arxivId(src = "") {
  if (!/arxiv\.org/.test(src)) return null;
  return src.match(/(\d{4}\.\d{4,5})(v\d+)?/)?.[1] || null;
}

export function folderName(paper) {
  const id = arxivId(paper.source?.value);
  return "paper-" + (id ? id.replace(".", "-") : paper.id);
}

export function selectCards(cardsJson, fileNames) {
  const have = new Set(fileNames);
  const ready = (cardsJson.cards || []).filter((c) => c.status === "ready" && have.has(`${c.slug}.html`));
  ready.sort((a, b) => (b.kind === "overview") - (a.kind === "overview"));
  return ready.map((c) => ({ slug: c.slug, title: c.title, file: `${c.slug}.html` }));
}

export function explainerJson(paper, cards) {
  const id = arxivId(paper.source?.value);
  const a = paper.authors || [];
  const who = a.length > 3 ? a.slice(0, 3).join(", ") + " et al." : a.join(", ");
  return {
    title: paper.title,
    description: [who, id && `arXiv ${id}`].filter(Boolean).join(" · "),
    date: (paper.addedAt || "").slice(0, 10),
    kind: "paper",
    source: id ? `https://arxiv.org/abs/${id}` : paper.source?.value || null,
    cards: cards.length,
  };
}
