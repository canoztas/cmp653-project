"""Tests for template extraction and matching."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dpdb.parser import parse_query
from dpdb.template import (
    extract_template,
    full_query_hash,
    is_exact_match,
    is_same_template,
    template_hash,
)


class TestTemplateExtraction:
    def test_same_query_same_template(self):
        q1 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        q2 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        assert extract_template(q1) == extract_template(q2)

    def test_different_literals_same_template(self):
        q1 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        q2 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'A'")
        assert is_same_template(q1, q2)

    def test_different_structure_different_template(self):
        q1 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        q2 = parse_query("SELECT SUM(l_quantity) FROM lineitem WHERE l_returnflag = 'R'")
        assert not is_same_template(q1, q2)

    def test_exact_match(self):
        q1 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        q2 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        assert is_exact_match(q1, q2)

    def test_not_exact_match_different_params(self):
        q1 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'R'")
        q2 = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_returnflag = 'A'")
        assert not is_exact_match(q1, q2)

    def test_template_hash_deterministic(self):
        q = parse_query("SELECT COUNT(*) FROM lineitem WHERE l_quantity > 10")
        t1 = template_hash(extract_template(q))
        t2 = template_hash(extract_template(q))
        assert t1 == t2
