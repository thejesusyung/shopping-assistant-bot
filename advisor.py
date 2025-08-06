"""
Streamlit UI for the shopping assistant chatbot.

Replaces the previous Gradio UI. Preserves language selection and RAG tool calling flow.
"""

import os
import json
import logging
from typing import Optional, List, Dict, Any

import streamlit as st
from openai import OpenAI
from pydantic import BaseModel, Field
from pydantic_core import PydanticOmit
from langchain_core.utils.function_calling import convert_to_openai_function

from prompts import (
    PRODUCT_SEARCH_SELECTION,
    PRODUCT_INFORMATION_DETAILS,
    PRODUCT_COMPARISON,
    GENERAL_ASSORTMENT_INQUIRY
)
from retrieval import ProductRetriever

# --- Intent Routing ---
from enum import Enum

class Intent(str, Enum):
    """Enumeration of user intents."""
    SEARCH_SELECTION = "product_search_selection"
    INFORMATION_DETAILS = "product_information_details"
    COMPARISON = "product_comparison"
    GENERAL_INQUIRY = "general_assortment_inquiry"

class IntentRouter(BaseModel):
    """Routes the user to the correct intent."""
    intent: Intent = Field(
        ...,
        description="The user's primary intent.",
    )

# --- Preference Extraction ---
from typing import Optional, List, Dict, Any, Literal
class UserPreference(BaseModel):
    """Model for user preferences."""
    brand: Optional[str] = Field(None, description="The user's preferred brand (e.g., 'Dell', 'HP', 'Apple').")
    min_ram_gb: Optional[int] = Field(None, description="The user's minimum required RAM in GB (e.g., 16).")
    min_storage_gb: Optional[int] = Field(None, description="The user's minimum required storage in GB (e.g., 512, 1024).")
    screen_inch: Optional[float] = Field(None, description="The user's preferred screen size in inches (e.g., 14.0, 15.6).")
    min_price: Optional[int] = Field(None, description="The user's minimum budget in USD.")
    max_price: Optional[int] = Field(None, description="The user's maximum budget in USD.")
    cpu_brand: Optional[Literal["Intel", "AMD", "Apple"]] = Field(None, description="The user's preferred CPU brand.")
    has_dedicated_gpu: Optional[bool] = Field(None, description="Whether the user requires a dedicated GPU (not integrated).")
    color: Optional[str] = Field(None, description="The user's preferred color for the product.")


class PreferenceExtractor(BaseModel):
    """Extracts user preferences from a conversation."""
    preference: UserPreference = Field(
        ...,
        description="The user's preferences.",
    )

# --- Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.propagate = False

if logger.hasHandlers():
    logger.handlers.clear()

file_handler = logging.FileHandler("chatbot.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --- Pydantic schema for tool calling ---
class ProductSearchTool(BaseModel):
    """Tool for searching products based on various filters."""
    query: Optional[str] = Field(None, description="A general search query for the product name, model, or keywords.")
    min_price: Optional[int] = Field(None, description="The minimum price of the product.")
    max_price: Optional[int] = Field(None, description="The maximum price of the product.")
    min_ram_gb: Optional[int] = Field(None, description="The minimum RAM size in GB.")
    min_storage_gb: Optional[int] = Field(None, description="The minimum storage size in GB.")
    brand: Optional[str] = Field(None, description="The brand of the product (e.g., 'Lenovo', 'Dell').")
    availability: Optional[List[str]] = Field(
        None, description="A list of availability statuses (e.g., 'in_stock', 'preorder')."
    )

# --- Core Advisor ---
class ShoppingAdvisor:
    def __init__(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        self.retriever = ProductRetriever()
        self.known_brands = self.retriever.get_all_brands()
        self.router_schema = convert_to_openai_function(IntentRouter)
        self.search_tool_schema = convert_to_openai_function(ProductSearchTool)
        self.preference_extractor_schema = convert_to_openai_function(PreferenceExtractor)

    def _extract_preferences(self, user_input: str) -> Optional[Dict[str, Any]]:
        """Extracts user preferences from user input using a structured format."""
        
        # Create a dynamic system prompt with the list of known brands
        brands_list = ", ".join(self.known_brands)
        system_prompt = f"""
You are a preference spotter. Your task is to identify and extract any product-related preferences the user states.

- **Extract from this list of known brands only**: {brands_list}.
- Other preferences can include RAM, storage, screen size, price, CPU (Intel, AMD, Apple), GPU, or color.
- Only extract preferences that are explicitly mentioned.
"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                tools=[{"type": "function", "function": self.preference_extractor_schema}],
                tool_choice={"type": "function", "function": {"name": "PreferenceExtractor"}},
                temperature=0.0,
            )
            tool_call = response.choices[0].message.tool_calls[0]
            args = json.loads(tool_call.function.arguments)
            preferences = args.get("preference", {})
            
            # Filter out any None values to only return explicitly stated preferences
            extracted_prefs = {k: v for k, v in preferences.items() if v is not None}
            
            if extracted_prefs:
                logger.info(f"Extracted preferences: {extracted_prefs}")
                return extracted_prefs
                
        except (json.JSONDecodeError, KeyError, ValueError, IndexError, AttributeError) as e:
            logger.warning(f"Preference extraction failed: {e}")
            return None
            
        return None

    def _get_intent(self, user_input: str) -> Intent:
        """Determines the user's intent with up to 5 retries."""
        messages = [
            {"role": "system", "content": "You are an intent classifier. Your task is to determine the user's primary goal."},
            {"role": "user", "content": user_input},
        ]
        
        for i in range(5):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4.1-mini",
                    messages=messages,
                    tools=[{"type": "function", "function": self.router_schema}],
                    tool_choice={"type": "function", "function": {"name": "IntentRouter"}},
                    temperature=0.0,
                )
                tool_call = response.choices[0].message.tool_calls[0]
                args = json.loads(tool_call.function.arguments)
                intent = Intent(args.get("intent"))
                logger.info(f"Intent classified as: {intent.value}")
                return intent
            except (json.JSONDecodeError, KeyError, ValueError, IndexError) as e:
                logger.warning(f"Intent classification failed on attempt {i+1}: {e}")
                continue

        logger.warning("Intent classification failed after 5 attempts. Defaulting to GENERAL_INQUIRY.")
        return Intent.GENERAL_INQUIRY

    def _get_language(self, user_input: str) -> Literal["English", "Russian"]:
        """Detects the language of the user's input."""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a language detector. Determine if the user's message is primarily in English or Russian. Respond with only 'English' or 'Russian'.",
                    },
                    {"role": "user", "content": user_input},
                ],
                max_tokens=5,
                temperature=0.0,
            )
            language = response.choices[0].message.content.strip()
            if language in ["English", "Russian"]:
                logger.info(f"Language detected: {language}")
                return language
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
        
        return "English"  # Default to English on failure

    def _execute_rag_flow(self, history: List[Dict[str, str]], system_prompt_content: str, preferences: Dict[str, Any], language: str) -> str:
        """Executes the full retrieval-augmented generation flow."""
        system_prompt = {"role": "system", "content": system_prompt_content}
        messages: List[Dict[str, Any]] = [system_prompt] + history
        logger.info("Messages to LLM (RAG flow):\n%s", json.dumps(messages, indent=2))

        try:
            first_response = self.client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=messages,
                tools=[{"type": "function", "function": self.search_tool_schema}],
                tool_choice="auto",
                temperature=0.0,
            )
            resp_msg = first_response.choices[0].message
            logger.info("LLM response (1st RAG call):\n%s", resp_msg.model_dump_json(indent=2))

            if getattr(resp_msg, "tool_calls", None):
                messages.append(resp_msg.model_dump())

                for tool_call in resp_msg.tool_calls:
                    args = json.loads(tool_call.function.arguments or "{}")
                    
                    # Merge stored preferences with tool call arguments
                    merged_args = preferences.copy()
                    merged_args.update(args)

                    logger.info("Tool call requested: %s with merged args:\n%s", tool_call.function.name, json.dumps(merged_args, indent=2))
                    
                    results = self.retriever.search_products(**merged_args)
                    products_json = json.dumps(results, indent=2, ensure_ascii=False)
                    logger.info("Retrieved products (JSON) for tool_call %s:\n%s", tool_call.id, products_json)

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": products_json,
                    })
                
                follow_up = f"Based on the tool results (JSON above), write a concise, helpful summary. Respond in {language}."
                messages.append({"role": "user", "content": follow_up})
                
                logger.info("Messages to LLM (2nd RAG call):\n%s", json.dumps(messages, indent=2))

                final_response = self.client.chat.completions.create(model="gpt-4.1-mini", messages=messages, temperature=0.0)
                final_content = (final_response.choices[0].message.content or "").strip()
                logger.info("Final LLM response: %s", final_content)
                return final_content

            return (resp_msg.content or "").strip()

        except Exception as e:
            logger.exception("Error in RAG flow: %s", e)
            return "Sorry, I encountered an error while processing your request."

    def _handle_comparison(self, history: List[Dict[str, str]], preferences: Dict[str, Any], language: str) -> str:
        """Handles a product comparison request using the RAG flow."""
        return self._execute_rag_flow(history, PRODUCT_COMPARISON["system_prompt"], preferences, language)

    def get_response(self, user_input: str, history: List[Dict[str, str]], preferences: Dict[str, Any]) -> str:
        """Routes the user to a specific handler based on the classified intent."""
        logger.info("User input: %s", user_input)
        language = self._get_language(user_input)
        intent = self._get_intent(user_input)
        truncated_history = history[-10:]

        if intent == Intent.SEARCH_SELECTION:
            return self._execute_rag_flow(truncated_history, PRODUCT_SEARCH_SELECTION["system_prompt"], preferences, language)
        elif intent == Intent.INFORMATION_DETAILS:
            return self._execute_rag_flow(truncated_history, PRODUCT_INFORMATION_DETAILS["system_prompt"], preferences, language)
        elif intent == Intent.COMPARISON:
            return self._execute_rag_flow(truncated_history, PRODUCT_COMPARISON["system_prompt"], preferences, language)
        else:
            return self._execute_rag_flow(truncated_history, GENERAL_ASSORTMENT_INQUIRY["system_prompt"], preferences, language)

def main():
    st.set_page_config(page_title="Future Tech - Shopping Assistant", page_icon="🛍️")
    st.title("🛍️ Future Tech — Shopping Assistant")

    api_key = st.session_state.get("openai_api_key")
    if not api_key:
        try:
            api_key = st.secrets["OPENAI_API_KEY"]
            st.session_state["openai_api_key"] = api_key
        except (KeyError, FileNotFoundError):
            st.warning("OpenAI API key not found in Streamlit secrets.")
            api_key_input = st.text_input("Please enter your OpenAI API Key:", type="password", key="api_key_input")
            if api_key_input:
                st.session_state["openai_api_key"] = api_key_input
                st.rerun()
            st.stop()
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "preferences" not in st.session_state:
        st.session_state.preferences = {}

    st.markdown("Welcome! How can I help you choose a laptop today? / Добро пожаловать! Чем я могу помочь вам в выборе ноутбука сегодня?")

    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    advisor = ShoppingAdvisor(api_key=api_key)

    prompt = st.chat_input("Type your message... / Введите ваше сообщение...")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        new_preferences = advisor._extract_preferences(prompt)
        if new_preferences:
            st.session_state.preferences.update(new_preferences)
            pref_list = [f"**{k.replace('_', ' ').title()}**: {v}" for k, v in new_preferences.items()]
            st.info(f"Preferences updated: {', '.join(pref_list)}")

        history = st.session_state.get("messages", [])
        preferences = st.session_state.get("preferences", {})
        reply = advisor.get_response(prompt, history, preferences)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    if st.button("Clear chat / Очистить чат", type="secondary"):
        st.session_state.messages = []
        st.session_state.preferences = {}
        st.rerun()

if __name__ == "__main__":
    main()