"""Querying graphs: the source's vertices/edges schema in sqlite3,
a 2-hop query both ways, and its Gremlin snippet rewritten as plain
Python over an adjacency dict, with traversers that carry a path.
Runs under python3 (3.10), standard library only.

The graph is TinkerPop's classic toy graph minus peter (5 vertices,
5 edges), so g.V.name starts marko, vadas as in the source.
"""
import json
import sqlite3

VERTICES = {1: {"name": "marko", "age": 29},
            2: {"name": "vadas", "age": 27},
            3: {"name": "lop", "lang": "java"},
            4: {"name": "josh", "age": 32},
            5: {"name": "ripple", "lang": "java"}}
# edge_id: (tail, head, label, weight)
EDGES = {9: (1, 3, "created", 0.4), 7: (1, 2, "knows", 0.5),
         8: (1, 4, "knows", 1.0), 10: (4, 5, "created", 1.0),
         11: (4, 3, "created", 0.4)}

SCHEMA = """
CREATE TABLE vertices (vertex_id integer PRIMARY KEY,
    properties json);
CREATE TABLE edges (edge_id integer PRIMARY KEY,
    tail_vertex integer REFERENCES vertices (vertex_id),
    head_vertex integer REFERENCES vertices (vertex_id),
    label text,
    properties json);
CREATE INDEX edges_tails ON edges (tail_vertex);
CREATE INDEX edges_heads ON edges (head_vertex);
"""

# 2 hops forward: what did the people marko knows create?
FORWARD = """
SELECT json_extract(v.properties, '$.name')
FROM edges e1
JOIN edges e2 ON e2.tail_vertex = e1.head_vertex
JOIN vertices v ON v.vertex_id = e2.head_vertex
WHERE e1.tail_vertex = 1 AND e1.label = 'knows'
  AND e2.label = 'created'
ORDER BY 1
"""

# 2 hops backward: who knows a creator of lop (vertex 3)?
BACKWARD = """
SELECT DISTINCT json_extract(v.properties, '$.name')
FROM edges e2
JOIN edges e1 ON e1.head_vertex = e2.tail_vertex
JOIN vertices v ON v.vertex_id = e1.tail_vertex
WHERE e2.head_vertex = 3 AND e2.label = 'created'
  AND e1.label = 'knows'
"""

# any number of hops: everything reachable from marko
REACH = """
WITH RECURSIVE r(id) AS (
  SELECT 1
  UNION
  SELECT e.head_vertex FROM edges e JOIN r ON e.tail_vertex = r.id)
SELECT count(*) - 1 FROM r
"""


def build():
    db = sqlite3.connect(":memory:")
    db.executescript(SCHEMA)
    db.executemany("INSERT INTO vertices VALUES (?, ?)",
                   [(i, json.dumps(p)) for i, p in VERTICES.items()])
    db.executemany(
        "INSERT INTO edges VALUES (?, ?, ?, ?, ?)",
        [(i, t, h, lab, json.dumps({"weight": w}))
         for i, (t, h, lab, w) in EDGES.items()])
    return db


def plan(db, sql):
    """Which index each table in the join is reached through."""
    return [row[3] for row in db.execute("EXPLAIN QUERY PLAN " + sql)]


# --- the Gremlin snippet, in plain Python -------------------------
# out_e[v] = v's outgoing edges, in insertion order (TinkerGraph's
# order for marko is e9, e7, e8, which is why the source shows 0.4
# then 0.5; never rely on it).
out_e = {v: [] for v in VERTICES}
for eid, (t, h, lab, w) in EDGES.items():
    out_e[t].append({"id": eid, "label": lab, "head": h,
                     "weight": w})


def V():                                   # g.V
    return list(VERTICES)


def names(vs):                             # .name
    return [VERTICES[v]["name"] for v in vs]


def outE(vs):                              # .outE
    return [e for v in vs for e in out_e[v]]


def weights(es):                           # .weight
    return [e["weight"] for e in es]


# --- traversers: the current object plus the path that led there --
def out(travs, label):                     # .out(label)
    return [(e["head"], path + [e["head"]])
            for v, path in travs for e in out_e[v]
            if e["label"] == label]


if __name__ == "__main__":
    db = build()
    print("forward: ", [r[0] for r in db.execute(FORWARD)])
    print("  plan:  ", plan(db, FORWARD))
    print("backward:", [r[0] for r in db.execute(BACKWARD)])
    print("  plan:  ", plan(db, BACKWARD))
    print("reachable from marko:", db.execute(REACH).fetchone()[0])

    print("g.V        ", V())
    print("g.V.name   ", names(V()))
    print("g.V.outE.weight", weights(outE(V())))

    travs = [(1, [1])]                     # g.V(1)
    for label in ("knows", "created"):
        travs = out(travs, label)
        print(f"out('{label}')", [names(p) for _, p in travs])
