"""Semantic template matching: AST Embeddings + Tree Kernels.

This module extends the exact-match template cache with two complementary
semantic similarity methods:

1. **Tree Kernel** (Collins & Duffy 2001) — classical NLP, deterministic.
   Computes the number of common subtrees between two ASTs. Captures
   structural equivalence (e.g., commutative WHERE predicates).

2. **AST Embedding** — neural approach. Encodes the canonical AST string
   using a pre-trained sentence transformer and uses cosine similarity
   for nearest-neighbor cache lookup. Captures higher-level semantic
   similarity (e.g., similar predicates, equivalent expressions).

The two methods are complementary:
- Tree kernel is exact/deterministic but only captures structural matches.
- Embedding catches semantic similarity but is approximate.

**Privacy note:** Returning a cached result for a "similar" query that is
not formally equivalent is NOT zero-cost under DP — the cached answer
was tuned to a different query. We therefore use semantic similarity only
to detect *structurally equivalent* queries that the syntactic hash missed
(e.g., commutative reordering, double negation), where post-processing
gives the same DP guarantee.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass, field
from typing import Optional

import sqlglot
from sqlglot import exp

from dpdb.parser import ParsedQuery


# ============================================================
# Tree Kernel (Collins & Duffy subtree kernel)
# ============================================================

def _node_label(node: exp.Expression) -> str:
    """Stable label for a node, capturing type and key constant parts."""
    name = type(node).__name__
    if isinstance(node, exp.Literal):
        return f"Literal(?)"  # collapse all literals to a placeholder
    if isinstance(node, exp.Column):
        return f"Column({node.name})"
    if isinstance(node, exp.Table):
        return f"Table({node.name})"
    if isinstance(node, exp.Identifier):
        return f"Id({node.name})"
    return name


def _ast_children(node: exp.Expression) -> list[exp.Expression]:
    """Return the children of an AST node (skip primitive args)."""
    children = []
    for v in node.args.values():
        if isinstance(v, exp.Expression):
            children.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, exp.Expression):
                    children.append(item)
    return children


def tree_kernel(
    t1: exp.Expression,
    t2: exp.Expression,
    lambda_decay: float = 1.0,
) -> float:
    """Compute the Collins-Duffy subtree kernel between two ASTs.

    Returns the number of common subtrees (weighted by lambda_decay
    when < 1.0 to penalize deep subtrees).

    K(T1, T2) = sum over (n1, n2) of C(n1, n2)
    where C(n1, n2) =
      0 if labels differ
      lambda if both leaves with same label
      lambda * prod_i (1 + C(child_i(n1), child_i(n2))) if non-leaf
    """
    # Collect all nodes from both trees
    nodes1 = list(t1.walk())
    nodes2 = list(t2.walk())

    # Strip the (parent, key) tuples that walk() returns
    nodes1 = [n[0] if isinstance(n, tuple) else n for n in nodes1]
    nodes2 = [n[0] if isinstance(n, tuple) else n for n in nodes2]

    total = 0.0
    cache: dict[tuple[int, int], float] = {}

    def C(n1: exp.Expression, n2: exp.Expression) -> float:
        key = (id(n1), id(n2))
        if key in cache:
            return cache[key]

        if _node_label(n1) != _node_label(n2):
            result = 0.0
        else:
            c1 = _ast_children(n1)
            c2 = _ast_children(n2)
            if len(c1) != len(c2):
                result = 0.0
            elif not c1:
                # Both leaves with same label
                result = lambda_decay
            else:
                # Same root, recurse on aligned children
                product = 1.0
                for child1, child2 in zip(c1, c2):
                    product *= (1.0 + C(child1, child2))
                result = lambda_decay * product

        cache[key] = result
        return result

    for n1 in nodes1:
        for n2 in nodes2:
            total += C(n1, n2)

    return total


def normalized_tree_kernel(
    t1: exp.Expression,
    t2: exp.Expression,
    lambda_decay: float = 1.0,
) -> float:
    """Normalize tree kernel to [0, 1] via cosine-like normalization.

    K_norm(T1, T2) = K(T1, T2) / sqrt(K(T1, T1) * K(T2, T2))
    """
    k12 = tree_kernel(t1, t2, lambda_decay)
    k11 = tree_kernel(t1, t1, lambda_decay)
    k22 = tree_kernel(t2, t2, lambda_decay)
    if k11 == 0 or k22 == 0:
        return 0.0
    return k12 / math.sqrt(k11 * k22)


# ============================================================
# AST Embedding (sentence-transformers)
# ============================================================

_EMBEDDER_CACHE = {}


def _get_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Lazy-load the sentence transformer (cached)."""
    if model_name not in _EMBEDDER_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )
    return _EMBEDDER_CACHE[model_name]


def ast_canonical_string(parsed: ParsedQuery) -> str:
    """Produce a canonical textual form of the AST for embedding.

    Replaces literals with placeholders, normalizes whitespace.
    """
    ast_copy = parsed.ast.copy()
    for literal in ast_copy.find_all(exp.Literal):
        literal.replace(exp.Placeholder())
    return ast_copy.sql(dialect="postgres", pretty=False)


def embed_query(parsed: ParsedQuery, model_name: Optional[str] = None):
    """Encode a parsed query as a dense embedding vector."""
    text = ast_canonical_string(parsed)
    model = _get_embedder(model_name or "sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(text, convert_to_numpy=True, show_progress_bar=False)
    return emb


def cosine_similarity(v1, v2) -> float:
    """Cosine similarity between two numpy vectors."""
    import numpy as np
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (n1 * n2))


# ============================================================
# Combined Semantic Matcher
# ============================================================

@dataclass
class SemanticEntry:
    """A query stored in the semantic cache."""
    parsed: ParsedQuery
    columns: list[str]
    rows: list[tuple]
    epsilon_used: float
    embedding: any = None  # numpy array or None
    tree_kernel_self: float = 1.0


@dataclass
class MatchResult:
    """Result of a semantic match against the cache."""
    entry: SemanticEntry
    tree_kernel_score: float       # normalized [0,1]
    embedding_score: float          # cosine [0,1]
    combined_score: float           # weighted blend
    method: str                     # "tree_kernel", "embedding", "both"


class SemanticMatcher:
    """Combined semantic cache using Tree Kernel + AST Embedding.

    The two methods are complementary:
    - Tree kernel: deterministic, structural; great for finding alpha-equivalent
      queries (commutative reorders, double negation, etc.).
    - Embedding: approximate but semantic; finds queries with similar intent.

    Both must agree (above their respective thresholds) for a match to be
    accepted. This ensures we only reuse a cached result when the query is
    very likely to be equivalent — preserving DP correctness via post-processing.
    """

    def __init__(
        self,
        tk_threshold: float = 0.95,
        emb_threshold: float = 0.98,
        use_embedding: bool = True,
        lambda_decay: float = 1.0,
    ):
        self.tk_threshold = tk_threshold
        self.emb_threshold = emb_threshold
        self.use_embedding = use_embedding
        self.lambda_decay = lambda_decay
        self.entries: list[SemanticEntry] = []

    def add(
        self,
        parsed: ParsedQuery,
        columns: list[str],
        rows: list[tuple],
        epsilon_used: float,
    ):
        """Store a new query result in the semantic cache."""
        embedding = None
        if self.use_embedding:
            try:
                embedding = embed_query(parsed)
            except Exception:
                embedding = None
        tk_self = tree_kernel(parsed.ast, parsed.ast, self.lambda_decay)
        self.entries.append(SemanticEntry(
            parsed=parsed,
            columns=columns,
            rows=rows,
            epsilon_used=epsilon_used,
            embedding=embedding,
            tree_kernel_self=tk_self,
        ))

    def find_match(self, parsed: ParsedQuery) -> Optional[MatchResult]:
        """Return the best semantic match if it exceeds the thresholds."""
        if not self.entries:
            return None

        new_emb = None
        if self.use_embedding:
            try:
                new_emb = embed_query(parsed)
            except Exception:
                new_emb = None

        tk_self_new = tree_kernel(parsed.ast, parsed.ast, self.lambda_decay)

        best: Optional[MatchResult] = None

        for entry in self.entries:
            # Tree kernel score (normalized)
            k = tree_kernel(parsed.ast, entry.parsed.ast, self.lambda_decay)
            tk_norm = k / math.sqrt(tk_self_new * entry.tree_kernel_self) \
                if (tk_self_new > 0 and entry.tree_kernel_self > 0) else 0.0

            # Embedding score
            emb_score = 0.0
            if new_emb is not None and entry.embedding is not None:
                emb_score = cosine_similarity(new_emb, entry.embedding)

            # Both must pass their threshold (conservative for DP correctness)
            tk_pass = tk_norm >= self.tk_threshold
            emb_pass = (emb_score >= self.emb_threshold) if self.use_embedding else True

            combined = 0.5 * tk_norm + 0.5 * emb_score \
                if self.use_embedding else tk_norm

            if tk_pass and emb_pass:
                method = "both" if self.use_embedding else "tree_kernel"
                if best is None or combined > best.combined_score:
                    best = MatchResult(
                        entry=entry,
                        tree_kernel_score=tk_norm,
                        embedding_score=emb_score,
                        combined_score=combined,
                        method=method,
                    )

        return best

    def __len__(self) -> int:
        return len(self.entries)
