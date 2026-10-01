"""Same schema, three meanings: "What was Q3 net revenue for EMEA?"

Three pipelines build three tables from the same raw orders. All
three expose the same columns (order_id, account_id, region,
amount), so a schema retriever cannot tell them apart.
Only the pipeline code says what a row is and what was dropped.

All orders, amounts and table names are illustrative (made up for
this card). Amounts are in thousands of dollars.
"""

# (order_id, account, region, gross, refunded, is_test, lines)
ORDERS = [
    (1, "A1", "EMEA", 120, 0, False, 2),
    (2, "A2", "EMEA", 80, 20, False, 1),
    (3, "A3", "EMEA", 200, 0, False, 3),
    (4, "T1", "EMEA", 50, 0, True, 1),   # internal test account
    (5, "A4", "EMEA", 150, 30, False, 2),
    (6, "A5", "NA", 300, 0, False, 1),   # other region
]


def fct_bookings(orders):
    """Gross bookings. Grain: one row per order. No exclusions."""
    return [(o[0], o[1], o[2], o[3]) for o in orders]


def fct_net_revenue(orders):
    """Net revenue. Grain: one row per order.
    amount = gross - refunds; internal test accounts dropped."""
    return [(o[0], o[1], o[2], o[3] - o[4])
            for o in orders if not o[5]]


def int_order_lines(orders):
    """Staging table. Grain: one row per order LINE. The pipeline
    copies the order's net amount onto every line."""
    rows = []
    for o in orders:
        if o[5]:
            continue
        rows += [(o[0], o[1], o[2], o[3] - o[4])] * o[6]
    return rows


def q3_emea(table):
    """The SQL an agent writes from the schema alone:
    SELECT SUM(amount) FROM t WHERE region = 'EMEA'."""
    return sum(r[3] for r in table if r[2] == "EMEA")


# A context card for the right table (fields from Part 4's list).
# Every fact carries where it came from.
CARD = {
    "table": "fct_net_revenue",
    "purpose": ("net revenue for reporting", "owner annotation"),
    "grain": ("one row per order", "dbt model SQL"),
    "key": ("order_id unique", "dbt test"),
    "filters": ("drops internal test accounts", "dbt model SQL"),
    "amount": ("gross minus refunds", "dbt model SQL"),
    "final_when": ("after finance reconciliation", "Airflow DAG"),
    "use_instead": ("fct_bookings for gross bookings", "owner"),
    "built_from_commit": ("a1b2c3d", "extractor"),
}


def card_is_stale(card, head_commit):
    """Refresh on change: a card built from an old commit of the
    pipeline code is stale documentation."""
    return card["built_from_commit"][0] != head_commit


if __name__ == "__main__":
    tables = {
        "fct_bookings": fct_bookings(ORDERS),
        "fct_net_revenue": fct_net_revenue(ORDERS),
        "int_order_lines": int_order_lines(ORDERS),
    }
    right = q3_emea(tables["fct_net_revenue"])
    for name, t in tables.items():
        ans = q3_emea(t)
        print(f"{name:16} rows={len(t):2}  Q3 EMEA = {ans:5}"
              f"  ({ans - right:+} vs net revenue)")
    print("card stale at head a1b2c3d?", card_is_stale(CARD, "a1b2c3d"))
    print("card stale at head 9f8e7d6?", card_is_stale(CARD, "9f8e7d6"))
