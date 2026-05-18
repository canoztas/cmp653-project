"""Define query workloads for benchmarking the DP middleware."""


def w1_repetitive(n: int = 20) -> list[str]:
    """W1: Same exact query repeated n times.
    Workload-aware should serve all but the first from cache.
    """
    q = "SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'"
    return [q] * n


def w2_parametric() -> list[str]:
    """W2: Same template, varying WHERE predicates (different nations).
    Same template but different parameters -> no exact cache hit,
    but demonstrates template tracking.
    """
    nations = [
        "FRANCE", "GERMANY", "UNITED STATES", "CHINA", "JAPAN",
        "BRAZIL", "INDIA", "RUSSIA", "UNITED KINGDOM", "CANADA",
        "EGYPT", "IRAN", "ALGERIA", "PERU", "INDONESIA",
        "FRANCE", "GERMANY", "UNITED STATES", "CHINA", "JAPAN",  # repeated
    ]
    return [
        f"SELECT COUNT(*) FROM customer WHERE c_mktsegment = 'AUTOMOBILE'"
        if i % 3 == 0 else
        f"SELECT COUNT(*) FROM orders WHERE o_orderstatus = '{'O' if i % 2 == 0 else 'F'}'"
        for i in range(20)
    ]


def w3_diverse() -> list[str]:
    """W3: Mix of COUNT/SUM/AVG on different tables.
    Diverse workload with little cache reuse opportunity.
    """
    return [
        "SELECT COUNT(*) FROM lineitem",
        "SELECT SUM(l_quantity) FROM lineitem",
        "SELECT AVG(l_extendedprice) FROM lineitem",
        "SELECT COUNT(*) FROM orders",
        "SELECT SUM(o_totalprice) FROM orders",
        "SELECT COUNT(*) FROM customer",
        "SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'",
        "SELECT SUM(l_quantity) FROM lineitem WHERE l_discount > 0.05",
        "SELECT AVG(l_extendedprice) FROM lineitem WHERE l_quantity < 25",
        "SELECT COUNT(*) FROM orders WHERE o_orderstatus = 'O'",
        "SELECT SUM(o_totalprice) FROM orders WHERE o_orderstatus = 'F'",
        "SELECT COUNT(*) FROM customer WHERE c_mktsegment = 'BUILDING'",
        "SELECT AVG(l_discount) FROM lineitem",
        "SELECT SUM(l_extendedprice) FROM lineitem WHERE l_returnflag = 'A'",
        "SELECT COUNT(*) FROM lineitem WHERE l_shipmode = 'AIR'",
        "SELECT AVG(l_quantity) FROM lineitem WHERE l_tax > 0.04",
        "SELECT SUM(l_extendedprice) FROM lineitem WHERE l_discount < 0.03",
        "SELECT COUNT(*) FROM orders WHERE o_orderpriority = '1-URGENT'",
        "SELECT AVG(o_totalprice) FROM orders",
        "SELECT COUNT(*) FROM supplier",
    ]


def w4_progressive_refinement() -> list[str]:
    """W4: Broad query followed by progressively narrower WHERE clauses.
    Tests whether broad results can serve narrower queries (post-processing).
    """
    return [
        # Broad
        "SELECT COUNT(*) FROM lineitem",
        "SELECT SUM(l_quantity) FROM lineitem",
        # Narrower
        "SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'",
        "SELECT SUM(l_quantity) FROM lineitem WHERE l_returnflag = 'R'",
        # Even narrower
        "SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R' AND l_linestatus = 'F'",
        "SELECT SUM(l_quantity) FROM lineitem WHERE l_returnflag = 'R' AND l_linestatus = 'F'",
        # Different branch
        "SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'A'",
        "SELECT SUM(l_quantity) FROM lineitem WHERE l_returnflag = 'A'",
        # Repeat some
        "SELECT COUNT(*) FROM lineitem",
        "SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'",
        "SELECT SUM(l_quantity) FROM lineitem",
        "SELECT SUM(l_quantity) FROM lineitem WHERE l_returnflag = 'A'",
        # More refinements
        "SELECT COUNT(*) FROM lineitem WHERE l_shipmode = 'AIR'",
        "SELECT COUNT(*) FROM lineitem WHERE l_shipmode = 'RAIL'",
        "SELECT SUM(l_extendedprice) FROM lineitem WHERE l_shipmode = 'AIR'",
        "SELECT SUM(l_extendedprice) FROM lineitem WHERE l_shipmode = 'RAIL'",
        # Repeats
        "SELECT COUNT(*) FROM lineitem WHERE l_shipmode = 'AIR'",
        "SELECT COUNT(*) FROM lineitem WHERE l_shipmode = 'RAIL'",
        "SELECT COUNT(*) FROM lineitem",
        "SELECT SUM(l_quantity) FROM lineitem",
    ]


ALL_WORKLOADS = {
    "W1_repetitive": w1_repetitive,
    "W2_parametric": w2_parametric,
    "W3_diverse": w3_diverse,
    "W4_progressive": w4_progressive_refinement,
}
