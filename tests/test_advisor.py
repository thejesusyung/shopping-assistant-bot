"""
This module contains unit tests for the ShoppingAdvisor class.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock

# Add the parent directory to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from advisor import ShoppingAdvisor, Intent, PRODUCT_SEARCH_SELECTION

@pytest.fixture
def advisor():
    """Provides a ShoppingAdvisor instance with a mocked OpenAI client."""
    with patch("advisor.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        advisor_instance = ShoppingAdvisor(api_key="test_key")
        advisor_instance.retriever = MagicMock()
        return advisor_instance

def test_preference_extraction(advisor):
    """Tests that brand preferences are correctly extracted from user input."""
    user_input = "I prefer Dell laptops"
    
    # Mock the response from the preference extractor LLM call
    mock_tool_call = MagicMock()
    mock_tool_call.function.arguments = '{"preference": {"brand": "Dell"}}'
    
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = [mock_tool_call]
    advisor.client.chat.completions.create.return_value = mock_response
    
    preferences = advisor._extract_preferences(user_input)
    
    advisor.client.chat.completions.create.assert_called_once()
    assert preferences is not None
    assert preferences.get("brand") == "Dell"

def test_get_response_applies_extracted_preferences(advisor):
    """
    Tests that an extracted brand preference is correctly applied in a subsequent search.
    """
    user_input = "Show me some options"
    history = [{"role": "user", "content": "I like AMD processors"}]
    preferences = {"brand": "AMD"}

    with patch.object(advisor, '_get_intent', return_value=Intent.SEARCH_SELECTION), \
         patch.object(advisor, '_execute_rag_flow') as mock_rag_flow:
        
        advisor.get_response(user_input, history, preferences)
        
        mock_rag_flow.assert_called_once()
        # Verify that the brand from preferences was passed to the RAG flow
        assert mock_rag_flow.call_args[0][2]['brand'] == "AMD"

def test_no_preference_extraction_when_not_stated(advisor):
    """Tests that no preference is extracted when none is stated."""
    user_input = "Just show me some laptops"
    
    # Simulate the LLM not returning a tool call for preferences
    mock_response = MagicMock()
    mock_response.choices[0].message.tool_calls = []
    advisor.client.chat.completions.create.return_value = mock_response
    
    preferences = advisor._extract_preferences(user_input)
    
    assert preferences is None

if __name__ == "__main__":
    pytest.main()
