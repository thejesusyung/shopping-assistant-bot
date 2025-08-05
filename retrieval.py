"""
This module provides the data retrieval logic for the shopping assistant.

It includes Pydantic models for data validation and a retriever class
to search for products in the loaded JSON data.
"""

import json
import logging
from typing import List, Optional, Dict, Any, Literal

from pydantic import BaseModel, Field, field_validator
from rapidfuzz import process, fuzz, distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProductVariant(BaseModel):
    """Pydantic model for a product variant."""
    sku: str
    ram_gb: int
    storage_gb: int
    cpu: str
    gpu: str
    screen_inch: float
    price_usd: int
    availability: Literal["in_stock", "out_of_stock", "limited", "preorder"]
    color: str

class Product(BaseModel):
    """Pydantic model for a product."""
    id: str
    brand: str
    model: str
    category: str
    keywords: List[str]
    aliases: List[str]
    variants: List[ProductVariant]

class ProductRetriever:
    """Handles loading and retrieving products from a JSON file."""

    def __init__(self, filepath: str = "products.json"):
        """
        Initializes the ProductRetriever.

        Args:
            filepath: Path to the JSON file containing product data.
        """
        self.products = self._load_products(filepath)
        self._build_search_index()

    def _load_products(self, filepath: str) -> List[Product]:
        """Loads products from a JSON file and validates them with Pydantic."""
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return [Product(**item) for item in data]
        except FileNotFoundError:
            logger.error(f"Product file not found at path: {filepath}")
            return []
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from file: {filepath}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading products: {e}")
            return []

    def _build_search_index(self):
        """Builds an index for fuzzy string matching."""
        self.search_choices = {}
        for product in self.products:
            key = f"{product.brand} {product.model}"
            texts_to_index = [key] + product.aliases
            self.search_choices[key] = " ".join(texts_to_index)

    def _is_match(self, query: str, product: Product, threshold: int = 2) -> bool:
        """
        Checks if a query matches a product's name or aliases using Levenshtein distance.

        A match is considered if the Levenshtein distance is less than or equal to the threshold.

        Args:
            query: The user's search query.
            product: The product to compare against.
            threshold: The maximum Levenshtein distance for a match.

        Returns:
            True if the query is a match, False otherwise.
        """
        query_lower = query.lower()
        
        # Check against the full model name
        if distance.Levenshtein.distance(query_lower, f"{product.brand.lower()} {product.model.lower()}") <= threshold:
            return True
        
        # Check against the model name
        if distance.Levenshtein.distance(query_lower, product.model.lower()) <= threshold:
            return True

        # Check against all aliases
        for alias in product.aliases:
            if distance.Levenshtein.distance(query_lower, alias.lower()) <= threshold:
                return True

        # Check against all keywords
        for keyword in product.keywords:
            if distance.Levenshtein.distance(query_lower, keyword.lower()) <= threshold:
                return True

        return False

    def search_products(
        self,
        query: Optional[str] = None,
        min_price: Optional[int] = None,
        max_price: Optional[int] = None,
        min_ram_gb: Optional[int] = None,
        min_storage_gb: Optional[int] = None,
        brand: Optional[str] = None,
        availability: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches and filters products based on various criteria.

        Args:
            query: A text query for fuzzy matching against product info.
            min_price: Minimum price in USD.
            max_price: Maximum price in USD.
            min_ram_gb: Minimum RAM in GB.
            min_storage_gb: Minimum storage in GB.
            brand: The product brand.
            availability: A list of availability statuses to filter by.

        Returns:
            A list of product variants matching the criteria.
        """
        filtered_products = self.products

        if query:
            filtered_products = [
                p for p in self.products if self._is_match(query, p)
            ]

        if brand:
            # Filter by brand using fuzzy matching
            brand_choices = list(set(p.brand for p in self.products))
            matched_brands = process.extract(brand, brand_choices, limit=3, score_cutoff=80)
            if matched_brands:
                best_brand = matched_brands[0][0]
                filtered_products = [p for p in filtered_products if p.brand.lower() == best_brand.lower()]


        results = []
        for product in filtered_products:
            for variant in product.variants:
                if min_price and variant.price_usd < min_price:
                    continue
                if max_price and variant.price_usd > max_price:
                    continue
                if min_ram_gb and variant.ram_gb < min_ram_gb:
                    continue
                if min_storage_gb and variant.storage_gb < min_storage_gb:
                    continue
                if availability and variant.availability not in availability:
                    continue
                
                result_item = product.model_dump(exclude={'variants'})
                result_item.update(variant.model_dump())
                results.append(result_item)
        
        # Sort results by price by default
        return sorted(results, key=lambda x: x["price_usd"])


if __name__ == "__main__":
    # Example usage of the retriever
    retriever = ProductRetriever()
    
    # 1. Search for a specific model
    print("--- Searching for 'ThinkBook 14 G3' ---")
    search_results = retriever.search_products(query="ThinkBook 14 G3")
    print(json.dumps(search_results, indent=2))

    # 2. Search with price and RAM filters
    print("\n--- Searching for laptops under $1500 with at least 16GB RAM ---")
    search_results = retriever.search_products(max_price=1500, min_ram_gb=16)
    print(json.dumps(search_results, indent=2))
    
    # 3. Fuzzy search for a misspelled model in Russian
    print("\n--- Fuzzy searching for 'Леново Тинкбук' ---")
    search_results = retriever.search_products(query="Леново Тинкбук")
    print(json.dumps(search_results, indent=2, ensure_ascii=False))

    # 4. Search by brand and availability
    print("\n--- Searching for available Dell laptops ---")
    search_results = retriever.search_products(brand="Dell", availability=["in_stock", "limited"])
    print(json.dumps(search_results, indent=2))
