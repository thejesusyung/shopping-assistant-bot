import pytest
from shopping_assistant.advisor import Advisor, ConversationState, Product, Variant

def test_advisor_smoke_test():
    """
    A basic smoke test to ensure the Advisor class can be instantiated
    and the basic components are importable.
    """
    try:
        # Test instantiation
        advisor = Advisor()
        assert advisor is not None
        assert isinstance(advisor.state, ConversationState)

        # Test data models
        variant = Variant(sku="TEST-01", price_usd=999, availability="in_stock")
        product = Product(
            id="TEST-PROD",
            brand="TestBrand",
            model="TestModel",
            category="test",
            keywords=["test"],
            aliases=["tst"],
            variants=[variant]
        )
        assert product.id == "TEST-PROD"
        assert variant.sku == "TEST-01"

    except ImportError as e:
        pytest.fail(f"Failed to import a component: {e}")
    except Exception as e:
        pytest.fail(f"An unexpected error occurred during the smoke test: {e}")

