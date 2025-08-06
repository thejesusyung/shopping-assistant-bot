"""
End-to-end test scenarios for the shopping assistant chatbot using an LLM evaluator.

This script runs a series of predefined conversational scenarios against the ShoppingAdvisor
to verify its core functionalities. Instead of simple keyword matching, it uses a powerful
LLM (gpt-4o-mini) to evaluate if the chatbot's response correctly addresses the user's
request in each step.

Usage:
    - Ensure the OPENAI_API_KEY environment variable is set.
    - Run the script from the root of the project: python test_scenarios.py
"""

import os
import json
import logging
import sys
from typing import List, Dict, Any
from openai import OpenAI

# Add the parent directory to the path so we can import advisor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advisor import ShoppingAdvisor, Intent

# --- Test Scenarios ---

TEST_SCENARIOS = [
    {
        "name": "Scenario 1: Product Search with Filters",
        "steps": [
            {
                "user_input": "Show me laptops with 32GB RAM cheaper than $2000",
                "expected_outcome": "Accepted outcome is a list of provided laptops that are under 2000$ and have 32GB RAM. We trust the values in the output, so we only check that the output doesn't have laptops that state they are under 32GB RAM or over2000$",
            }
        ],
    },
    {
        "name": "Scenario 2: Contextual Comparison",
        "steps": [
            {
                "user_input": "Show me laptops from Dell and HP",
                "expected_outcome": "The assistant must list available laptops from both Dell and HP, specifically mentioning 'Dell XPS 13' and 'HP Envy 15'.",
            },
            {
                "user_input": "Compare the Dell XPS 13 with the HP Envy 15",
                "expected_outcome": "The assistant must provide a side-by-side comparison of the 'Dell XPS 13' and 'HP Envy 15', focusing on key specs like CPU, RAM, storage, and price.",
            },
        ],
    },
    {
        "name": "Scenario 3: Typo Tolerance",
        "steps": [
            {
                "user_input": "Tell me more about the 'ThinkBok Pro'",
                "expected_outcome": "The assistant must provide details for the 'Lenovo ThinkBook 14 G3'.",
            }
        ],
    },
    {
        "name": "Scenario 4: User Preference Handling (AMD)",
        "steps": [
            {
                "user_input": "I prefer AMD processors.",
                "expected_outcome": "Assistant should preferably acknowledge the user's preference for AMD processors and show available AMD laptops with Ryzen processors. However, this is a pass if there are no contradictions. Important that assistant does not recommend a non AMD laptop.",
            },
            {
                "user_input": "Show me some powerful laptops.",
                "expected_outcome": "The assistant must recommend powerful laptops that specifically have AMD CPUs (e.g., models with 'Ryzen' processors), correctly applying the previously stated preference.",
            },
        ],
    },
    {
        "name": "Scenario 5: Multi-turn Context & Language Adherence",
        "steps": [
            {
                "user_input": "Show me all Dell laptops.",
                "expected_outcome": "The assistant must list available Dell laptops, including the 'XPS 13' and 'Inspiron 16'.",
            },
            {
                "user_input": "Which one has more storage?",
                "expected_outcome": "The assistant must correctly identify from the previous context that both the 'XPS 13' and the 'Inspiron 16' have the highest storage capacity (1TB) and state this.",
            },
            {
                "user_input": "And what about the price of the Inspiron?",
                "expected_outcome": "The assistant must state the price of the 'Dell Inspiron 16' (the 32GB RAM version), which is $1499.",
            },
        ],
    },
]

# --- Test Runner & LLM-based Evaluator ---

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def evaluate_step(history: List[Dict[str, str]], latest_response: str, expected_outcome: str, client: OpenAI) -> bool:
    """
    Uses a powerful LLM to evaluate if the chatbot's response meets the test criteria.
    """
    system_prompt = """
You are an intelligent test evaluator. Your goal is to assess if the assistant's response
semantically satisfies the user's request, based on the conversation history and an
expected outcome. Be flexible with phrasing.

**Instructions:**
1.  **Understand the User's Goal**: From the `Conversation History` and the `Expected Outcome`, determine the core requirement of the user's last message.
2.  **Assess the Assistant's Response**: Read the `Assistant's Latest Response`. Does it successfully address the user's goal?
3.  **Focus on Facts, Not Phrasing**: The exact wording is not important. The key is whether the information provided is correct and complete according to the `Expected Outcome`. Minor phrasing differences or responding in a different language than the request are acceptable as long as the core information is correct.
4.  **Make a decision**:
    - If the response contains the correct key information and addresses the user's goal, respond with **'PASSED'**.
    - If the response is factually incorrect, misses key information mentioned in the `Expected Outcome`, or completely fails to address the user's goal, respond with **'FAILED'**.

**Your response must be a single word: either `PASSED` or `FAILED`. Do not provide any explanation.**
"""

    # Format the history for the evaluator prompt
    formatted_history = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in history])
    
    user_prompt = f"""
**Conversation History:**
```json
{json.dumps(history, indent=2)}
```

**Assistant's Latest Response:**
```
{latest_response}
```

**Expected Outcome:**
`{expected_outcome}`

---
Based on all the provided information, did the assistant's response meet the expected outcome?
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=10,
            temperature=0.0,
        )
        result = response.choices[0].message.content.strip().upper()
        return result == "PASSED"
    except Exception as e:
        print(f"{bcolors.FAIL}Evaluator call failed: {e}{bcolors.ENDC}")
        return False


def run_test_scenarios():
    """
    Runs all defined test scenarios and reports the results using an LLM evaluator.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print(f"{bcolors.FAIL}ERROR: OPENAI_API_KEY environment variable not set.{bcolors.ENDC}")
        return

    client = OpenAI(api_key=api_key)
    advisor = ShoppingAdvisor(api_key=api_key)
    total_passed = 0
    total_failed = 0

    for scenario in TEST_SCENARIOS:
        print(f"\n{bcolors.HEADER}--- Running Scenario: {scenario['name']} ---{bcolors.ENDC}")
        
        history: List[Dict[str, str]] = []
        preferences: Dict[str, Any] = {}
        scenario_passed_all_steps = True

        for i, step in enumerate(scenario["steps"]):
            user_input = step['user_input']
            expected_outcome = step['expected_outcome']
            print(f"{bcolors.OKBLUE}Step {i+1}: User input: \"{user_input}\"{bcolors.ENDC}")

            # Update preferences from the new input
            new_preferences = advisor._extract_preferences(user_input)
            if new_preferences:
                preferences.update(new_preferences)

            # Add user input to history BEFORE getting response
            history.append({"role": "user", "content": user_input})

            # Get the response
            response = advisor.get_response(user_input, history, preferences)
            print(f"{bcolors.OKCYAN}Assistant response: \"{response}\"{bcolors.ENDC}")
            
            # Update history with assistant response
            history.append({"role": "assistant", "content": response})

            # Evaluate the step using the LLM evaluator
            is_passed = evaluate_step(history, response, expected_outcome, client)
            
            if not is_passed:
                print(f"{bcolors.FAIL}Step {i+1} FAILED. The response did not meet the expected outcome.{bcolors.ENDC}")
                scenario_passed_all_steps = False
                break
            else:
                print(f"{bcolors.OKGREEN}Step {i+1} PASSED.{bcolors.ENDC}")

        if scenario_passed_all_steps:
            total_passed += 1
            print(f"{bcolors.OKGREEN}--- Scenario '{scenario['name']}' PASSED ---\n{bcolors.ENDC}")
        else:
            total_failed += 1
            print(f"{bcolors.FAIL}--- Scenario '{scenario['name']}' FAILED ---\n{bcolors.ENDC}")

    print(f"\n{bcolors.BOLD}--- Test Summary ---{bcolors.ENDC}")
    print(f"{bcolors.OKGREEN}Passed: {total_passed}{bcolors.ENDC}")
    print(f"{bcolors.FAIL}Failed: {total_failed}{bcolors.ENDC}")

if __name__ == "__main__":
    # Suppress verbose logging from the advisor during tests
    logging.getLogger("advisor").setLevel(logging.WARNING)
    run_test_scenarios()
