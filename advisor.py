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

# --- Logging ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
# Prevent logs from propagating to the root logger, which Streamlit might configure for console output
logger.propagate = False

# Remove any existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Create a file handler to write logs to a file
file_handler = logging.FileHandler("chatbot.log", mode="w")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# --- Config ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

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
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.retriever = ProductRetriever()
        self.router_schema = convert_to_openai_function(IntentRouter)
        self.search_tool_schema = convert_to_openai_function(ProductSearchTool)

    def _get_intent(self, user_input: str) -> Intent:
        """Determines the user's intent with up to 5 retries."""
        messages = [
            {"role": "system", "content": "You are an intent classifier. Your task is to determine the user's primary goal."},
            {"role": "user", "content": user_input},
        ]
        
        for i in range(5):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4.1-nano",
                    messages=messages,
                    tools=[{"type": "function", "function": self.router_schema}],
                    tool_choice={"type": "function", "function": {"name": "IntentRouter"}},
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

    def _execute_rag_flow(self, history: List[Dict[str, str]], system_prompt_content: str) -> str:
        """Executes the full retrieval-augmented generation flow."""
        system_prompt = {"role": "system", "content": system_prompt_content}
        messages: List[Dict[str, Any]] = [system_prompt] + history
        logger.info("Messages to LLM (RAG flow):\n%s", json.dumps(messages, indent=2))

        try:
            # 1) Ask the model to call the product search tool
            first_response = self.client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=messages,
                tools=[{"type": "function", "function": self.search_tool_schema}],
                tool_choice="auto",
            )
            resp_msg = first_response.choices[0].message
            logger.info("LLM response (1st RAG call):\n%s", resp_msg.model_dump_json(indent=2))

            # 2) If tool is called, execute it and get a final summary
            if getattr(resp_msg, "tool_calls", None):
                messages.append(resp_msg.model_dump())  # Append assistant's reply with tool call

                for tool_call in resp_msg.tool_calls:
                    args = json.loads(tool_call.function.arguments or "{}")
                    logger.info("Tool call requested: %s with args:\n%s", tool_call.function.name, json.dumps(args, indent=2))
                    
                    results = self.retriever.search_products(**args)
                    products_json = json.dumps(results, indent=2, ensure_ascii=False)
                    logger.info("Retrieved products (JSON) for tool_call %s:\n%s", tool_call.id, products_json)

                    messages.append({
                        "tool_call_id": tool_call.id,
                        "role": "tool",
                        "name": tool_call.function.name,
                        "content": products_json,
                    })
                
                # Use a generic follow-up prompt
                follow_up = "Based on the tool results (JSON above), write a concise, helpful summary. Respond in the user's language."
                messages.append({"role": "user", "content": follow_up})
                
                logger.info("Messages to LLM (2nd RAG call):\n%s", json.dumps(messages, indent=2))

                final_response = self.client.chat.completions.create(model="gpt-4.1-nano", messages=messages)
                final_content = (final_response.choices[0].message.content or "").strip()
                logger.info("Final LLM response: %s", final_content)
                return final_content

            # 3) If no tool is called, return the model's direct response
            return (resp_msg.content or "").strip()

        except Exception as e:
            logger.exception("Error in RAG flow: %s", e)
            return "Sorry, I encountered an error while processing your request."

    def _handle_comparison(self, history: List[Dict[str, str]]) -> str:
        """Handles a product comparison request."""
        system_prompt = {"role": "system", "content": PRODUCT_COMPARISON["system_prompt"]}
        messages: List[Dict[str, Any]] = [system_prompt] + history
        logger.info("Messages to LLM (Comparison flow):\n%s", json.dumps(messages, indent=2))
        
        try:
            response = self.client.chat.completions.create(model="gpt-4.1-nano", messages=messages)
            content = (response.choices[0].message.content or "").strip()
            logger.info("LLM response (Comparison): %s", content)
            return content
        except Exception as e:
            logger.exception("Error in comparison flow: %s", e)
            return "Sorry, I couldn't process the comparison request."

    def get_response(self, user_input: str, history: List[Dict[str, str]]) -> str:
        """Routes the user to a specific handler based on the classified intent."""
        logger.info("User input: %s", user_input)
        intent = self._get_intent(user_input)

        if intent == Intent.SEARCH_SELECTION:
            return self._execute_rag_flow(history, PRODUCT_SEARCH_SELECTION["system_prompt"])
        elif intent == Intent.INFORMATION_DETAILS:
            return self._execute_rag_flow(history, PRODUCT_INFORMATION_DETAILS["system_prompt"])
        elif intent == Intent.COMPARISON:
            return self._handle_comparison(history)
        else:  # GENERAL_INQUIRY is the fallback
            return self._execute_rag_flow(history, GENERAL_ASSORTMENT_INQUIRY["system_prompt"])


# --- Streamlit App ---
def main():
    st.set_page_config(page_title="Future Tech - Shopping Assistant", page_icon="🛍️")
    st.title("🛍️ Future Tech — Shopping Assistant")

    # Intro
    st.markdown("Welcome! How can I help you choose a laptop today? / Добро пожаловать! Чем я могу помочь вам в выборе ноутбука сегодня?")

    # Session state for chat history
    if "messages" not in st.session_state:
        st.session_state.messages: List[Dict[str, str]] = []

    # Show history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    advisor = ShoppingAdvisor()

    # Chat input
    prompt = st.chat_input("Type your message... / Введите ваше сообщение...")
    if prompt:
        # Echo user
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get assistant response
        history = st.session_state.get("messages", [])
        reply = advisor.get_response(prompt, history)
        st.session_state.messages.append({"role": "assistant", "content": reply})
        with st.chat_message("assistant"):
            st.markdown(reply)

    # Clear button
    if st.button("Clear chat / Очистить чат", type="secondary"):
        st.session_state.messages = []
        st.rerun()

if __name__ == "__main__":
    main()
