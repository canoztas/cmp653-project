"""Tests for semantic matching (Tree Kernel + AST Embedding)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.parser import parse_query
from dpdb.semantic import (
    SemanticMatcher,
    ast_canonical_string,
    cosine_similarity,
    normalized_tree_kernel,
    tree_kernel,
)


class TestTreeKernel:
    def test_self_kernel_positive(self):
        q = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        k = tree_kernel(q.ast, q.ast)
        assert k > 0

    def test_identical_queries_have_max_normalized_kernel(self):
        q1 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        q2 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        nk = normalized_tree_kernel(q1.ast, q2.ast)
        assert nk == pytest.approx(1.0, abs=1e-6)

    def test_different_literals_high_similarity(self):
        """Queries with different WHERE literals should be highly similar
        because literals are collapsed to placeholders."""
        q1 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        q2 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'A'")
        nk = normalized_tree_kernel(q1.ast, q2.ast)
        assert nk > 0.95

    def test_different_aggregates_lower_similarity(self):
        q1 = parse_query("SELECT COUNT(*) FROM lineitem")
        q2 = parse_query("SELECT SUM(l_quantity) FROM lineitem")
        nk = normalized_tree_kernel(q1.ast, q2.ast)
        assert 0 < nk < 1.0

    def test_different_tables_lower_similarity(self):
        q1 = parse_query("SELECT COUNT(*) FROM lineitem")
        q2 = parse_query("SELECT COUNT(*) FROM orders")
        nk = normalized_tree_kernel(q1.ast, q2.ast)
        assert nk < 1.0


class TestCanonicalString:
    def test_replaces_literals(self):
        q = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        s = ast_canonical_string(q)
        assert "'R'" not in s

    def test_same_template_same_string(self):
        q1 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        q2 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'A'")
        assert ast_canonical_string(q1) == ast_canonical_string(q2)


class TestSemanticMatcher:
    def test_empty_matcher_no_match(self):
        m = SemanticMatcher(use_embedding=False)
        q = parse_query("SELECT COUNT(*) FROM lineitem")
        assert m.find_match(q) is None

    def test_exact_query_matches_itself(self):
        m = SemanticMatcher(use_embedding=False)
        q = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        m.add(q, ["count"], [(42,)], 1.0)
        result = m.find_match(q)
        assert result is not None
        assert result.tree_kernel_score == pytest.approx(1.0, abs=1e-6)

    def test_template_match_via_tree_kernel(self):
        """Same template (different literals) should match via tree kernel."""
        m = SemanticMatcher(tk_threshold=0.95, use_embedding=False)
        q1 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        q2 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'A'")
        m.add(q1, ["count"], [(42,)], 1.0)
        # Different literal but same structure -> should match
        result = m.find_match(q2)
        assert result is not None

    def test_different_query_no_match(self):
        m = SemanticMatcher(tk_threshold=0.99, use_embedding=False)
        q1 = parse_query("SELECT COUNT(*) FROM lineitem")
        q2 = parse_query("SELECT SUM(l_quantity) FROM orders")
        m.add(q1, ["count"], [(42,)], 1.0)
        result = m.find_match(q2)
        assert result is None


class TestCosineSimilarity:
    def test_identical_vectors(self):
        import numpy as np
        v = np.array([1.0, 2.0, 3.0])
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self):
        import numpy as np
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_zero_vector(self):
        import numpy as np
        v1 = np.array([0.0, 0.0])
        v2 = np.array([1.0, 1.0])
        assert cosine_similarity(v1, v2) == 0.0
