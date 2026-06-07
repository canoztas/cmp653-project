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


class TestExactCacheKeyNoCollision:
    """Regression: the param hash must NOT collide on literals that flatten to the
    same naive delimiter-join, or on a number vs. its string form."""

    def test_no_collision_on_delimiter_in_literal(self):
        # 'a|b'+'c'  vs  'a'+'b|c'  both naive-join to "a|b|c" -> must still differ
        a = parse_query("SELECT COUNT(*) FROM adult WHERE education = 'a|b' AND sex = 'c'")
        b = parse_query("SELECT COUNT(*) FROM adult WHERE education = 'a' AND sex = 'b|c'")
        assert full_query_hash(a) != full_query_hash(b)
        assert not is_exact_match(a, b)

    def test_number_and_string_literal_differ(self):
        a = parse_query("SELECT COUNT(*) FROM adult WHERE age = 1")
        b = parse_query("SELECT COUNT(*) FROM adult WHERE age = '1'")
        assert full_query_hash(a) != full_query_hash(b)

    def test_exact_repeat_still_matches(self):
        a = parse_query("SELECT COUNT(*) FROM adult WHERE education = 'a' AND sex = 'b|c'")
        b = parse_query("SELECT COUNT(*) FROM adult WHERE education = 'a' AND sex = 'b|c'")
        assert full_query_hash(a) == full_query_hash(b)
        assert is_exact_match(a, b)
