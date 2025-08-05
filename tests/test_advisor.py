"""
This module contains unit tests for the ShoppingAdvisor class.
"""

import pytest
from unittest.mock import patch, MagicMock
from advisor import ShoppingAdvisor, Intent, PRODUCT_SEARCH_SELECTION, PRODUCT_COMPARISON

@pytest.fixture
def advisor():
    """Provides a ShoppingAdvisor instance with a mocked OpenAI client."""
    with patch("advisor.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        advisor_instance = ShoppingAdvisor(api_key="test_key")
        advisor_instance.retriever = MagicMock()
        return advisor_instance

def test_get_response_routes_to_search_selection(advisor):
    """Tests that a search query correctly routes to the RAG flow."""
    user_input = "Show me laptops under $1500"
    with patch.object(advisor, '_get_intent', return_value=Intent.SEARCH_SELECTION) as mock_get_intent, \
         patch.object(advisor, '_execute_rag_flow') as mock_rag_flow:
        
        mock_rag_flow.return_value = "Success"
        
        response = advisor.get_response(user_input, [], {})
        
        mock_get_intent.assert_called_once_with(user_input)
        mock_rag_flow.assert_called_once_with([], PRODUCT_SEARCH_SELECTION["system_prompt"], {})
        assert response == "Success"

def test_get_response_routes_to_comparison(advisor):
    """Tests that a comparison query correctly routes to the comparison handler."""
    user_input = "Compare Dell and HP"
    with patch.object(advisor, '_get_intent', return_value=Intent.COMPARISON) as mock_get_intent, \
         patch.object(advisor, '_execute_rag_flow') as mock_rag_flow:

        mock_rag_flow.return_value = "Comparison success"

        response = advisor.get_response(user_input, [], {})
        
        mock_get_intent.assert_called_once_with(user_input)
        mock_rag_flow.assert_called_once_with([], PRODUCT_COMPARISON["system_prompt"], {})
        assert response == "Comparison success"

def test_user_preferences_are_applied(advisor):
    """Tests that user preferences are correctly applied to the search."""
    user_input = "Show me laptops"
    preferences = {"brand": "AMD"} # Simulating a user preference for AMD
    with patch.object(advisor, '_get_intent', return_value=Intent.SEARCH_SELECTION), \
         patch.object(advisor, '_execute_rag_flow') as mock_rag_flow:
        
        advisor.get_response(user_input, [], preferences)
        
        mock_rag_flow.assert_called_once()
        # Check that the brand preference was passed to the RAG flow
        assert mock_rag_flow.call_args[0][2]['brand'] == "AMD"

if __name__ == "__main__":
    pytest.main()
