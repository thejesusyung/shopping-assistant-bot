# -*- coding: utf-8 -*-
"""
End-to-end тестовые сценарии для чат-бота‑помощника по покупкам с использованием LLM‑оценщика.

Скрипт запускает серию заранее определённых диалоговых сценариев против ShoppingAdvisor
для проверки его ключевых функций. Вместо простого сопоставления по ключевым словам
используется мощная LLM‑модель (gpt-4.1), которая оценивает, корректно ли ответил
чат‑бот на запрос пользователя на каждом шаге.

Использование:
    - Убедитесь, что задана переменная окружения OPENAI_API_KEY.
    - Запустите из корня проекта: python test_scenarios.py
"""

import os
import json
import logging
import sys
from typing import List, Dict, Any
from openai import OpenAI

# Добавляем родительскую папку в путь, чтобы импортировать advisor
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from advisor import ShoppingAdvisor, Intent

# --- Тестовые сценарии ---

TEST_SCENARIOS = [
    {
        "name": "Сценарий 1: Поиск товаров с фильтрами",
        "steps": [
            {
                "user_input": "Покажи ноутбуки с 32 ГБ ОЗУ дешевле $2000",
                "expected_outcome": "Приемлемым результатом является список предоставленных ноутбуков, которые стоят менее 2000 долларов и имеют 32 ГБ ОЗУ. Мы доверяем значениям в выводе, если ОЗУ не указано, то мы не проверяем этот параметр. Поэтому мы только проверяем, что в выводе нет ноутбуков, которые указывают, что они имеют менее 32 ГБ ОЗУ или стоят более 2000 долларов",
            }
        ],
    },
    {
        "name": "Сценарий 2: Контекстное сравнение",
        "steps": [
            {
                "user_input": "Покажи ноутбуки от Dell и HP",
                "expected_outcome": (
                    "Ассистент должен перечислить доступные ноутбуки от Dell и HP, "
                    "явно упомянув ‘Dell XPS 13’ и ‘HP Envy 15’."
                ),
            },
            {
                "user_input": "Сравни Dell XPS 13 и HP Envy 15",
                "expected_outcome": (
                    "Ассистент должен предоставить сравнение ‘Dell XPS 13’ и ‘HP Envy 15’ "
                    "по ключевым характеристикам: CPU, ОЗУ, хранилище и цена."
                ),
            },
        ],
    },
    {
        "name": "Сценарий 3: Терпимость к опечаткам",
        "steps": [
            {
                "user_input": "Расскажи подробнее про ‘ThinkBok Pro’",
                "expected_outcome": (
                    "Ассистент должен предоставить сведения по ‘Lenovo ThinkBook 14 G3’."
                ),
            }
        ],
    },
    {
        "name": "Сценарий 4: Учёт пользовательских предпочтений (AMD)",
        "steps": [
            {
                "user_input": "Я предпочитаю процессоры AMD.",
                "expected_outcome": (
                    "Ассистент должен признать предпочтение AMD и показать доступные ноутбуки "
                    "с процессорами Ryzen. Однако сценарий считается пройденным, если нет "
                    "противоречий. Важно, чтобы ассистент не рекомендовал ноутбук не на AMD."
                ),
            },
            {
                "user_input": "Покажи мощные ноутбуки.",
                "expected_outcome": (
                    "Ассистент должен рекомендовать мощные ноутбуки именно с процессорами AMD "
                    "(например, на Ryzen), корректно применив ранее указанное предпочтение."
                ),
            },
        ],
    },
    {
        "name": "Сценарий 5: Мультиходовой контекст и соблюдение языка",
        "steps": [
            {
                "user_input": "Покажи все ноутбуки Dell.",
                "expected_outcome": (
                    "Ассистент должен перечислить доступные ноутбуки Dell, включая ‘XPS 13’ и ‘Inspiron 16’."
                ),
            },
            {
                "user_input": "У какого больше хранилище?",
                "expected_outcome": (
                    "Ассистент должен корректно определить из предыдущего контекста, что и ‘XPS 13’, и ‘Inspiron 16’ "
                    "имеют максимальную ёмкость накопителя (1 ТБ), и явно сообщить это."
                ),
            },
            {
                "user_input": "А сколько стоит Inspiron?",
                "expected_outcome": (
                    "Ассистент должен указать цену ‘Dell Inspiron 16’ (версия с 32 ГБ ОЗУ) — $1499."
                ),
            },
        ],
    },
]

# --- Запуск тестов и LLM‑оценщик ---

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
    Использует мощную LLM, чтобы оценить, соответствует ли ответ чат-бота критериям теста.
    """
    system_prompt = """
Ты — умный тестовый оценщик. Твоя цель — определить, удовлетворяет ли ответ ассистента
семантически запрос пользователя, исходя из истории разговора и ожидаемого результата.
Будь гибким к формулировкам.

**Инструкции:**
1. **Пойми цель пользователя**: по Истории разговора и Ожидаемому результату определи основное требование последнего сообщения пользователя.
2. **Оцени ответ ассистента**: прочитай Последний ответ ассистента. Удовлетворяет ли он цели пользователя?
3. **Сфокусируйся на фактах, а не на формулировках**: точная словесная форма не важна. Ключевое — правильность и полнота сведений относительно Ожидаемого результата. Небольшие различия в формулировках или ответ на другом языке допустимы, если основная информация корректна.
4. **Прими решение**:
   - Если ответ содержит верную ключевую информацию и решает задачу пользователя, ответь **"PASSED"**.
   - Если ответ фактически неверен, упускает ключевую информацию из Ожидаемого результата или не решает задачу пользователя, ответь **"FAILED"**.

**Твой ответ должен быть одним словом: PASSED или FAILED. Никаких объяснений.**
"""

    # Форматируем историю для запроса к оценщику
    formatted_history = "\n".join([f"- {msg['role']}: {msg['content']}" for msg in history])

    user_prompt = f"""
**История разговора:**
json
{json.dumps(history, indent=2)}


**Последний ответ ассистента:**
{latest_response}


**Ожидаемый результат:**
{expected_outcome}

---
На основании всей предоставленной информации, соответствует ли ответ ассистента ожидаемому результату?
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4.1",
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
    Запускает все определённые сценарии и сообщает результаты при помощи LLM‑оценщика.
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

            # Обновляем предпочтения из нового ввода
            new_preferences = advisor._extract_preferences(user_input)
            if new_preferences:
                preferences.update(new_preferences)

            # Добавляем пользовательский ввод в историю ДО получения ответа
            history.append({"role": "user", "content": user_input})

            # Получаем ответ ассистента
            response = advisor.get_response(user_input, history, preferences)
            print(f"{bcolors.OKCYAN}Assistant response: \"{response}\"{bcolors.ENDC}")

            # Обновляем историю ответом ассистента
            history.append({"role": "assistant", "content": response})

            # Оцениваем шаг при помощи LLM‑оценщика
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
    # Подавляем подробные логи advisor во время тестов
    logging.getLogger("advisor").setLevel(logging.WARNING)
    run_test_scenarios()
