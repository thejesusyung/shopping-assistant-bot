"""
This module contains unit tests for the ShoppingAdvisor class.
"""

import pytest
from unittest.mock import patch, MagicMock
from advisor import ShoppingAdvisor, Intent

@pytest.fixture
def advisor():
    """Provides a ShoppingAdvisor instance with a mocked OpenAI client."""
    with patch("advisor.OpenAI") as MockOpenAI:
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        # Pass a dummy API key, as it's now required by the constructor
        advisor_instance = ShoppingAdvisor(api_key="test_key")
        advisor_instance.retriever = MagicMock()
        return advisor_instance

def mock_openai_response(intent: str):
    """Factory to create a mock OpenAI response for intent classification."""
    return MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    tool_calls=[
                        MagicMock(
                            function=MagicMock(
                                arguments=f'{{"intent": "{intent}"}}'
                            )
                        )
                    ]
                )
            )
        ]
    )

def test_intent_classification_search_selection(advisor):
    """Tests that a search query is correctly classified."""
    user_input = "Show me laptops under $1500"
    advisor.client.chat.completions.create.return_value = mock_openai_response(Intent.SEARCH_SELECTION.value)
    
    intent = advisor._get_intent(user_input)
    assert intent == Intent.SEARCH_SELECTION

def test_intent_classification_information_details(advisor):
    """Tests that a product detail query is correctly classified."""
    user_input = "Tell me more about the Dell XPS"
    advisor.client.chat.completions.create.return_value = mock_openai_response(Intent.INFORMATION_DETAILS.value)

    intent = advisor._get_intent(user_input)
    assert intent == Intent.INFORMATION_DETAILS

def test_intent_classification_comparison(advisor):
    """Tests that a comparison query is correctly classified."""
    user_input = "Compare the Dell XPS and the MacBook Air"
    advisor.client.chat.completions.create.return_value = mock_openai_response(Intent.COMPARISON.value)

    intent = advisor._get_intent(user_input)
    assert intent == Intent.COMPARISON

def test_intent_classification_general_inquiry(advisor):
    """Tests that a general query is correctly classified."""
    user_input = "What brands do you have?"
    advisor.client.chat.completions.create.return_value = mock_openai_response(Intent.GENERAL_INQUIRY.value)

    intent = advisor._get_intent(user_input)
    assert intent == Intent.GENERAL_INQUIRY

def test_intent_classification_fallback(advisor):
    """Tests that the router falls back to GENERAL_INQUIRY after 5 failed attempts."""
    user_input = "some gibberish"
    # Simulate an error during API call
    advisor.client.chat.completions.create.side_effect = ValueError("Invalid response")

    intent = advisor._get_intent(user_input)
    assert intent == Intent.GENERAL_INQUIRY
    # Check that it was called 5 times
    assert advisor.client.chat.completions.create.call_count == 5

def test_get_response_routes_to_search_selection(advisor):
    """Tests that the router correctly calls the RAG flow for a search query."""
    user_input = "Show me laptops under $1500"
    with patch.object(advisor, '_get_intent', return_value=Intent.SEARCH_SELECTION) as mock_get_intent, \
         patch.object(advisor, '_execute_rag_flow') as mock_rag_flow:
        advisor.get_response(user_input, [])
        mock_get_intent.assert_called_once_with(user_input)
        mock_rag_flow.assert_called_once()

def test_get_response_routes_to_information_details(advisor):
    """Tests that the router correctly calls the RAG flow for a details query."""
    user_input = "Tell me more about the Dell XPS"
    with patch.object(advisor, '_get_intent', return_value=Intent.INFORMATION_DETAILS) as mock_get_intent, \
         patch.object(advisor, '_execute_rag_flow') as mock_rag_flow:
        advisor.get_response(user_input, [])
        mock_get_intent.assert_called_once_with(user_input)
        mock_rag_flow.assert_called_once()

def test_get_response_routes_to_comparison(advisor):
    """Tests that the router correctly calls the comparison handler."""
    user_input = "Compare Dell and HP"
    with patch.object(advisor, '_get_intent', return_value=Intent.COMPARISON) as mock_get_intent, \
         patch.object(advisor, '_handle_comparison') as mock_comparison:
        advisor.get_response(user_input, [])
        mock_get_intent.assert_called_once_with(user_input)
        mock_comparison.assert_called_once()

def test_get_response_routes_to_general_inquiry(advisor):
    """Tests that the router correctly calls the RAG flow for a general query."""
    user_input = "What do you sell?"
    with patch.object(advisor, '_get_intent', return_value=Intent.GENERAL_INQUIRY) as mock_get_intent, \
         patch.object(advisor, '_execute_rag_flow') as mock_rag_flow:
        advisor.get_response(user_input, [])
        mock_get_intent.assert_called_once_with(user_input)
        mock_rag_flow.assert_called_once()

if __name__ == "__main__":
    pytest.main()
