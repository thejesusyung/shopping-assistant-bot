# retrieval_v2.py
"""
Variant-level, flexible product retriever for the shopping assistant.

Key improvements:
- Filters at the VARIANT level (brand/price/RAM/CPU/GPU live on variants).
- Brand-only and price-only searches work.
- max_price is inclusive (≤), e.g., "under 1500" returns 1499 items.
- Free-text `query` is a SOFT scorer (ranks) rather than a hard filter.
- Sensible availability defaults (in_stock, limited, preorder).
- Gentle fallback: relax price by +10% if nothing found (keeps brand/availability).

Usage:
    from retrieval_v2 import ProductRetriever
    r = ProductRetriever("products.json")
    r.search_products(brand="Dell", limit=10)
    r.search_products(max_price=1500, sort_by="price_asc")
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path


DATA_PATH = Path(__file__).with_name("products.json")


@dataclass
class VariantRow:
    product_id: str
    brand: str
    model: str
    category: str
    sku: str
    ram_gb: int
    storage_gb: int
    storage_type: str
    weight_kg: float
    cpu: str
    gpu: str
    screen_inch: float
    price_usd: float
    availability: str
    color: str


def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _cpu_brand(cpu: str) -> Optional[str]:
    s = _norm(cpu)
    if not s:
        return None
    if "intel" in s:
        return "intel"
    if "amd" in s:
        return "amd"
    if "apple" in s or s.startswith("m") or "m1" in s or "m2" in s or "m3" in s:
        return "apple"
    return None


def _flatten(products: List[Dict[str, Any]]) -> List[VariantRow]:
    rows: List[VariantRow] = []
    for p in products:
        for v in p.get("variants", []):
            rows.append(
                VariantRow(
                    product_id=p["id"],
                    brand=p["brand"],
                    model=p["model"],
                    category=p.get("category", ""),
                    sku=v["sku"],
                    ram_gb=v.get("ram_gb"),
                    storage_gb=v.get("storage_gb"),
                    storage_type=v.get("storage_type", ""),
                    weight_kg=v.get("weight_kg"),
                    cpu=v.get("cpu", ""),
                    gpu=v.get("gpu", ""),
                    screen_inch=v.get("screen_inch"),
                    price_usd=float(v.get("price_usd")),
                    availability=v.get("availability", ""),
                    color=v.get("color", ""),
                )
            )
    return rows


class ProductRetriever:
    def __init__(self, data_path: Optional[str] = None):
        path = Path(data_path) if data_path else DATA_PATH
        with open(path, "r", encoding="utf-8") as f:
            products = json.load(f)
        self.rows: List[VariantRow] = _flatten(products)

    def get_all_brands(self) -> List[str]:
        """Returns a sorted list of unique brand names."""
        return sorted(list(set(r.brand for r in self.rows)))

    def search_products(
        self,
        query: Optional[str] = None,
        brand: Optional[str] = None,
        max_price: Optional[float] = None,
        min_price: Optional[float] = None,
        min_ram_gb: Optional[int] = None,
        cpu_brand: Optional[str] = None,
        gpu: Optional[str] = None,
        availability: Optional[List[str]] = None,
        category: Optional[str] = None,
        limit: int = 12,
        sort_by: str = "relevance",  # "price_asc" | "price_desc"
    ) -> List[Dict[str, Any]]:
        rows = list(self.rows)

        # Default availability: items a user can realistically buy soon
        if availability is None:
            availability = ["in_stock", "limited", "preorder"]
        allow = {a.lower() for a in availability}
        rows = [r for r in rows if _norm(r.availability) in allow]

        if category:
            rows = [r for r in rows if _norm(r.category) == _norm(category)]

        if brand:
            b = _norm(brand)
            rows = [r for r in rows if _norm(r.brand) == b]

        if min_price is not None:
            rows = [r for r in rows if r.price_usd >= float(min_price)]
        if max_price is not None:
            # inclusive: "under 1500" returns 1499 etc.
            rows = [r for r in rows if r.price_usd <= float(max_price)]

        if min_ram_gb is not None:
            rows = [r for r in rows if r.ram_gb >= int(min_ram_gb)]

        if cpu_brand:
            cb = _norm(cpu_brand)
            rows = [r for r in rows if _cpu_brand(r.cpu) == cb]

        if gpu:
            g = _norm(gpu)
            if g in ("rtx", "nvidia", "discrete", "dedicated"):
                rows = [r for r in rows if _norm(r.gpu) and _norm(r.gpu) != "integrated"]
            else:
                rows = [r for r in rows if g in _norm(r.gpu)]

        # Free-text query as a *soft* scorer (no hard filtering)
        if query:
            q = _norm(query)

            def score(r: VariantRow) -> int:
                s = 0
                tokens = [q] + q.split()
                hay = f"{r.brand} {r.model} {r.sku}".lower()
                for t in tokens:
                    if t and t in hay:
                        s += 1
                return s

            rows_scored = [(score(r), r) for r in rows]
            # Keep all rows, but rank by score desc then price asc
            rows = [r for _, r in sorted(rows_scored, key=lambda x: (-x[0], x[1].price_usd))]

        # Sorting
        if sort_by == "price_asc":
            rows.sort(key=lambda r: (r.price_usd, r.brand, r.model, r.ram_gb))
        elif sort_by == "price_desc":
            rows.sort(key=lambda r: (-r.price_usd, r.brand, r.model, r.ram_gb))

        # Fallback: nothing found → relax price by +10% (keep brand & availability)
        if not rows:
            expanded = list(self.rows)
            expanded = [r for r in expanded if _norm(r.availability) in allow]
            if brand:
                expanded = [r for r in expanded if _norm(r.brand) == _norm(brand)]
            if max_price is not None:
                cap = float(max_price) * 1.10
                expanded = [r for r in expanded if r.price_usd <= cap]
            rows = expanded

        # Format output
        out: List[Dict[str, Any]] = []
        for r in rows[:limit]:
            out.append(asdict(r))
        return out


if __name__ == "__main__":
    r = ProductRetriever()
    print("Dell (brand-only):", len(r.search_products(brand="Dell", limit=20)))
    print("Under 1500:", len(r.search_products(max_price=1500, sort_by="price_asc", limit=50)))
