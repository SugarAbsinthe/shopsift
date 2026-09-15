"""Hybrid product retrieval with independently ranked recall channels."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional

import chromadb
from sentence_transformers import SentenceTransformer

from src.retrieval.models import (
    ProductCandidate,
    RetrievalConstraints,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStats,
)
from src.retrieval.index_manifest import resolve_collection_names


RETRIEVAL_ALGORITHM_VERSION = "hybrid-rrf-v2-evidence"
RRF_K = 60


def reciprocal_rank_fusion(
    ranked_ids: dict[str, list[int]], rrf_k: int = RRF_K
) -> tuple[list[int], dict[int, float], dict[int, dict[str, int]]]:
    """Fuse rank-only lists without comparing incompatible distance scales."""
    scores: dict[int, float] = defaultdict(float)
    source_ranks: dict[int, dict[str, int]] = defaultdict(dict)
    for source, product_ids in ranked_ids.items():
        seen: set[int] = set()
        for rank, product_id in enumerate(product_ids, start=1):
            if product_id in seen:
                continue
            seen.add(product_id)
            scores[product_id] += 1.0 / (rrf_k + rank)
            source_ranks[product_id][source] = rank
    ordered = sorted(scores, key=lambda pid: (-scores[pid], pid))
    return ordered, dict(scores), dict(source_ranks)


class ProductRetriever:
    """Retrieve products through description, spec, and FTS5 channels."""

    def __init__(
        self,
        chroma_dir: str,
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        catalog_db: str | None = None,
        cache=None,
        *,
        model=None,
        client=None,
    ):
        self.model = model or SentenceTransformer(embedding_model)
        self.client = client or chromadb.PersistentClient(path=chroma_dir)
        collection_names, self.index_version = resolve_collection_names(chroma_dir)
        self.desc_col = self.client.get_collection(collection_names["descriptions"])
        self.spec_col = self.client.get_collection(collection_names["specs"])
        self.review_col = self.client.get_collection(collection_names["reviews"])
        self.catalog_db = catalog_db
        self._cache = cache
        self._sparse_conn: sqlite3.Connection | None = None
        self._sparse_available = False
        self._initialize_sparse_index()

    def _initialize_sparse_index(self) -> None:
        """Build a process-local FTS index; never mutate the catalog database."""
        if not self.catalog_db or not Path(self.catalog_db).is_file():
            return
        try:
            catalog = sqlite3.connect(
                f"file:{Path(self.catalog_db).resolve()}?mode=ro", uri=True
            )
            catalog.row_factory = sqlite3.Row
            rows = catalog.execute(
                "SELECT product_id, name, brand, category, subcategory, "
                "description, specs FROM products ORDER BY product_id"
            ).fetchall()
            catalog.close()
            sparse = sqlite3.connect(":memory:")
            sparse.execute(
                "CREATE VIRTUAL TABLE product_fts USING fts5("
                "product_id UNINDEXED, searchable, tokenize='trigram')"
            )
            sparse.executemany(
                "INSERT INTO product_fts(product_id, searchable) VALUES (?, ?)",
                [
                    (
                        row["product_id"],
                        " ".join(
                            str(row[key] or "")
                            for key in (
                                "name", "brand", "category", "subcategory",
                                "description", "specs",
                            )
                        ),
                    )
                    for row in rows
                ],
            )
            self._sparse_conn = sparse
            self._sparse_available = True
        except (OSError, sqlite3.Error):
            self._sparse_conn = None
            self._sparse_available = False

    @staticmethod
    def _metadata_ids(results: dict[str, Any]) -> list[int]:
        metadatas = results.get("metadatas") or []
        if not metadatas or not metadatas[0]:
            return []
        product_ids: list[int] = []
        for metadata in metadatas[0]:
            try:
                product_ids.append(int(metadata["product_id"]))
            except (KeyError, TypeError, ValueError):
                continue
        return product_ids

    def _vector_recall(
        self, collection, query_embedding: list[float], limit: int
    ) -> list[int]:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=limit,
            include=["metadatas"],
        )
        return self._metadata_ids(results)

    def _sparse_recall(self, query: str, limit: int) -> tuple[list[int], bool]:
        if not self._sparse_available or self._sparse_conn is None:
            return [], True
        user_query = query.splitlines()[0]
        raw_terms = re.findall(r"[\u4e00-\u9fff]+|[A-Za-z0-9][A-Za-z0-9._-]*", user_query)
        terms: list[str] = []
        for term in raw_terms:
            if re.fullmatch(r"[\u4e00-\u9fff]+", term) and len(term) > 3:
                terms.extend(term[index : index + 3] for index in range(len(term) - 2))
            elif len(term) >= 3:
                terms.append(term)
        terms = list(dict.fromkeys(terms))[:20]
        if not terms:
            return [], False
        expression = " OR ".join(
            f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms
        )
        try:
            rows = self._sparse_conn.execute(
                "SELECT product_id FROM product_fts WHERE searchable MATCH ? "
                "ORDER BY bm25(product_fts) LIMIT ?",
                (expression, limit),
            ).fetchall()
            return [int(row[0]) for row in rows], False
        except sqlite3.Error:
            return [], True

    def _load_catalog_products(self, product_ids: list[int]) -> dict[int, ProductCandidate]:
        if not product_ids or not self.catalog_db:
            return {}
        placeholders = ",".join("?" for _ in product_ids)
        try:
            conn = sqlite3.connect(
                f"file:{Path(self.catalog_db).resolve()}?mode=ro", uri=True
            )
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT product_id, name, brand, category, subcategory, price, "
                "rating, sales_count, release_date, description, specs "
                f"FROM products WHERE product_id IN ({placeholders})",
                product_ids,
            ).fetchall()
            conn.close()
        except (OSError, sqlite3.Error):
            return {}

        products: dict[int, ProductCandidate] = {}
        for row in rows:
            try:
                specs = json.loads(row["specs"]) if row["specs"] else {}
            except (TypeError, json.JSONDecodeError):
                specs = {}
            product = ProductCandidate(
                product_id=row["product_id"],
                name=row["name"] or "",
                brand=row["brand"] or "",
                category=row["category"] or "",
                subcategory=row["subcategory"] or "",
                price=row["price"],
                rating=row["rating"],
                sales_count=row["sales_count"],
                release_date=row["release_date"] or "",
                description=row["description"] or "",
                specs=specs,
            )
            products[product.product_id] = product
        return products

    @staticmethod
    def _matches_constraints(
        product: ProductCandidate, constraints: RetrievalConstraints
    ) -> bool:
        if constraints.min_price is not None and (
            product.price is None or product.price < constraints.min_price
        ):
            return False
        if constraints.max_price is not None and (
            product.price is None or product.price > constraints.max_price
        ):
            return False
        if constraints.category and product.category.casefold() != constraints.category.casefold():
            return False
        excluded = {brand.casefold() for brand in constraints.excluded_brands}
        if product.brand.casefold() in excluded:
            return False
        return True

    @staticmethod
    def _rerank_bonus(
        product: ProductCandidate, query: str, constraints: RetrievalConstraints
    ) -> float:
        normalized_query = re.sub(r"\s+", "", query).casefold()
        name = re.sub(r"\s+", "", product.name).casefold()
        searchable_specs = re.sub(
            r"\s+", "", json.dumps(product.specs, ensure_ascii=False)
        ).casefold()
        bonus = 0.0
        if len(normalized_query) >= 3 and normalized_query in name:
            bonus += 0.02
        compact_terms = [
            term.casefold()
            for term in re.findall(r"[A-Za-z]+\s*\d+[A-Za-z0-9-]*", query)
        ]
        if any(re.sub(r"\s+", "", term) in searchable_specs for term in compact_terms):
            bonus += 0.01
        preferred = {brand.casefold() for brand in constraints.preferred_brands}
        if product.brand.casefold() in preferred:
            bonus += 0.005
        return bonus

    def _retrieve_reviews(
        self, query_embedding: list[float], product_ids: list[int], top_k: int
    ) -> dict[int, list[dict[str, str]]]:
        if not product_ids:
            return {}
        try:
            results = self.review_col.query(
                query_embeddings=[query_embedding],
                n_results=max(top_k * 3, 1),
                where={"product_id": {"$in": product_ids[:10]}},
                include=["documents", "metadatas"],
            )
        except Exception:
            return {}
        reviews: dict[int, list[dict[str, str]]] = defaultdict(list)
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        for document, metadata in zip(documents, metadatas):
            try:
                product_id = int(metadata["product_id"])
            except (KeyError, TypeError, ValueError):
                continue
            reviews[product_id].append({
                "aspect": str(metadata.get("aspect", "")),
                "sentiment": str(metadata.get("sentiment", "")),
                "content": str(document),
            })
        return dict(reviews)

    def search(
        self,
        query: str | RetrievalQuery,
        top_k: int = 5,
        filters: Optional[dict] = None,
    ) -> RetrievalResult:
        """Return structured, auditable retrieval results."""
        started = time.perf_counter()
        request = query if isinstance(query, RetrievalQuery) else RetrievalQuery(
            text=query, top_k=top_k, constraints=filters or {}
        )
        overfetch = min(request.top_k * 3, 60)
        query_embedding = self.model.encode(request.text).tolist()
        description_ids = self._vector_recall(self.desc_col, query_embedding, overfetch)
        spec_ids = self._vector_recall(self.spec_col, query_embedding, overfetch)
        sparse_ids, sparse_fallback = self._sparse_recall(request.text, overfetch)

        ranked_ids = {
            "description": description_ids,
            "spec": spec_ids,
            "sparse": sparse_ids,
        }
        ordered_ids, scores, source_ranks = reciprocal_rank_fusion(ranked_ids)
        catalog_products = self._load_catalog_products(ordered_ids)
        filtered_count = 0
        candidates: list[ProductCandidate] = []
        for product_id in ordered_ids:
            product = catalog_products.get(product_id)
            if product is None or not self._matches_constraints(product, request.constraints):
                filtered_count += 1
                continue
            ranks = source_ranks[product_id]
            product.sources = list(ranks)
            product.source_ranks = ranks
            product.score = scores[product_id] + self._rerank_bonus(
                product, request.text, request.constraints
            )
            candidates.append(product)

        candidates.sort(key=lambda item: (-item.score, item.product_id))
        products = candidates[: request.top_k]
        product_ids = [product.product_id for product in products]
        reviews = self._retrieve_reviews(query_embedding, product_ids, request.top_k)
        stats = RetrievalStats(
            source_hits={source: len(set(ids)) for source, ids in ranked_ids.items()},
            fused_candidates=len(ordered_ids),
            filtered_candidates=filtered_count,
            returned_candidates=len(products),
            sparse_fallback=sparse_fallback,
            index_version=self.index_version,
            duration_ms=round((time.perf_counter() - started) * 1000),
        )
        from backend.logging_config import record_retrieval_stats

        record_retrieval_stats(stats)
        evidence = self._build_evidence(products, stats.index_version)
        return RetrievalResult(
            products=products,
            reviews_by_product=reviews,
            stats=stats,
            constraints=request.constraints,
            evidence=evidence,
        )

    @staticmethod
    def _build_evidence(
        products: list[ProductCandidate], index_version: str
    ) -> list:
        """Build stable catalog facts without copying user query text."""
        from src.retrieval.models import EvidenceItem

        evidence: list[EvidenceItem] = []
        for product in products:
            source_rank = min(product.source_ranks.values()) if product.source_ranks else None
            values = {
                "product_id": product.product_id,
                "price": product.price,
                "brand": product.brand,
                "category": product.category,
            }
            for field, value in values.items():
                if value is None or value == "":
                    continue
                evidence.append(EvidenceItem(
                    evidence_id=f"catalog:{index_version}:{product.product_id}:{field}",
                    product_id=product.product_id,
                    field=field,
                    value=value,
                    source_rank=source_rank,
                    index_version=index_version,
                ))
        return evidence

    def retrieve_result(
        self, query: str, top_k: int = 5, filters: Optional[dict] = None
    ) -> RetrievalResult:
        """Return structured results while preserving the existing cache path."""
        cache_args = (
            query,
            top_k,
            filters,
            self.index_version,
            RETRIEVAL_ALGORITHM_VERSION,
        )
        if self._cache is not None:
            cached = self._cache.get(*cache_args)
            if cached is not None:
                try:
                    result = RetrievalResult.model_validate(json.loads(cached))
                except (TypeError, ValueError, json.JSONDecodeError):
                    result = None
                if result is not None:
                    from backend.logging_config import log, mark_cache_hit

                    mark_cache_hit()
                    log("rag_cache_hit")
                    return result

        result = self.search(query, top_k=top_k, filters=filters)
        if self._cache is not None:
            self._cache.set(
                query,
                top_k,
                json.dumps(result.model_dump(), ensure_ascii=False, separators=(",", ":")),
                filters,
                self.index_version,
                RETRIEVAL_ALGORITHM_VERSION,
            )
        return result

    def retrieve(
        self, query: str, top_k: int = 5, filters: Optional[dict] = None
    ) -> str:
        """Compatibility API used by the Agent and search tool."""
        result = self.retrieve_result(query, top_k=top_k, filters=filters)
        formatted = self._format_for_prompt(result.products, result.reviews_by_product)
        return formatted

    def format_result(self, result: RetrievalResult) -> str:
        """Format a structured result for the legacy prompt context."""
        return self._format_for_prompt(result.products, result.reviews_by_product)

    def _format_for_prompt(
        self,
        products: list[ProductCandidate],
        reviews_by_pid: dict[int, list[dict[str, str]]],
    ) -> str:
        lines = ["## 相关产品推荐\n"]
        if not products:
            lines.append("未找到满足当前约束的商品。")
            return "\n".join(lines)
        for index, product in enumerate(products, start=1):
            lines.append(f"### {index}. {product.name or '?'}")
            lines.append(
                f"- 品牌: {product.brand or '?'} | 品类: "
                f"{product.category or '?'} / {product.subcategory or '?'}"
            )
            lines.append(
                f"- 价格: RMB{product.price if product.price is not None else '?'} | "
                f"评分: {product.rating if product.rating is not None else '?'} 分 | "
                f"销量: {product.sales_count or 0}"
            )
            lines.append(
                f"- 发布日期: {product.release_date or '?'} | 产品ID: {product.product_id}"
            )
            if product.description:
                lines.append(f"- 产品简介: {product.description}")
            if product.specs:
                lines.append("- 规格参数:")
                lines.extend(f"  {key}: {value}" for key, value in product.specs.items())
            if product.product_id in reviews_by_pid:
                lines.append("- 用户评价摘要:")
                for review in reviews_by_pid[product.product_id][:3]:
                    marker = "+" if review["sentiment"] == "positive" else "-"
                    lines.append(f"  [{marker}] {review['aspect']}: {review['content']}")
            lines.append("")
        return "\n".join(lines)
