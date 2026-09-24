"""The MapReduce programming model, run on one machine.

The user writes only map and reduce; run_mapreduce plays the
library: it calls map on every input, groups the intermediate
pairs by key (the shuffle), then calls reduce once per key in
increasing key order. Python 3.10, standard library only.
"""
from collections import defaultdict

DOCS = {"d1": "the cat sat",
        "d2": "the cat ate the fish",
        "d3": "a dog sat"}


def run_mapreduce(mapper, reducer, inputs, trace=False):
    groups = defaultdict(list)             # the shuffle
    for k1, v1 in inputs.items():
        pairs = list(mapper(k1, v1))       # map: (k1, v1) -> [(k2, v2)]
        if trace:
            print(f"  map {k1}: {pairs}")
        for k2, v2 in pairs:
            groups[k2].append(v2)
    out = {}
    for k2 in sorted(groups):              # reduce sees keys in order
        if trace:
            print(f"  reduce {k2!r} <- {groups[k2]}")
        out[k2] = reducer(k2, iter(groups[k2]))
    return out


# Word count: the source's pseudocode, in Python
def wc_map(doc_name, text):
    for w in text.split():
        yield w, 1                         # EmitIntermediate(w, "1")


def wc_reduce(word, counts):
    return sum(counts)                     # Emit(result)


# Inverted index: map emits (word, doc id), reduce sorts the ids
def ii_map(doc_name, text):
    for w in set(text.split()):            # once per doc
        yield w, doc_name


def ii_reduce(word, doc_ids):
    return sorted(doc_ids)


def count_pairs(mapper):
    return sum(len(list(mapper(k, v))) for k, v in DOCS.items())


if __name__ == "__main__":
    print("word count")
    wc = run_mapreduce(wc_map, wc_reduce, DOCS, trace=True)
    print(" ", wc)
    print("inverted index")
    ii = run_mapreduce(ii_map, ii_reduce, DOCS)
    print(" ", ii)
    print("intermediate pairs: word count",
          count_pairs(wc_map), "| inverted index",
          count_pairs(ii_map), "| distinct keys", len(wc))
    # check against a one-machine answer
    from collections import Counter
    words = " ".join(DOCS.values()).split()
    assert wc == dict(Counter(words))
    assert ii["the"] == ["d1", "d2"] and ii["sat"] == ["d1", "d3"]
    print("matches collections.Counter: True")
