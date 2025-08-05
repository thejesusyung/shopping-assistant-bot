# Shopping Assistant Bot

This project is a conversational shopping assistant that helps users choose products based on their preferences. It understands natural language queries, handles typos, and maintains conversation context.

## Features

- **Natural Language Understanding**: Powered by OpenAI's GPT models to understand user queries.
- **Product Filtering**: Supports filtering by price, brand, specs, and availability.
- **Typo Correction**: Uses Levenshtein distance to handle typos in product names.
- **Contextual Conversations**: Maintains a short-term memory of the conversation.
- **Multilingual**: Responds in both English and Russian.

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd shopping-assistant-bot
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up your OpenAI API Key:**
    The application uses Streamlit's secrets management. For local development, you can create a `.streamlit/secrets.toml` file in the project root:
    ```toml
    # .streamlit/secrets.toml
    OPENAI_API_KEY = "your-openai-api-key"
    ```
    Alternatively, the application will prompt you to enter the API key in the UI if it's not found.

## How to Run

To start the Streamlit application, run the following command in the project root:

```bash
streamlit run advisor.py
```

The application will be available at `http://localhost:8501`.

---

# Бот-помощник для покупок

Этот проект представляет собой разговорного помощника по покупкам, который помогает пользователям выбирать товары в соответствии с их предпочтениями. Он понимает запросы на естественном языке, исправляет опечатки и поддерживает контекст разговора.

## Возможности

- **Понимание естественного языка**: Использует модели GPT от OpenAI для понимания запросов пользователей.
- **Фильтрация продуктов**: Поддерживает фильтрацию по цене, бренду, характеристикам и наличию.
- **Исправление опечаток**: Использует расстояние Левенштейна для исправления опечаток в названиях продуктов.
- **Контекстные диалоги**: Сохраняет краткосрочную память о разговоре.
- **Многоязычность**: Отвечает на английском и русском языках.

## Установка

1.  **Клонируйте репозиторий:**
    ```bash
    git clone <repository-url>
    cd shopping-assistant-bot
    ```

2.  **Создайте и активируйте виртуальное окружение:**
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Установите зависимости:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Настройте ваш OpenAI API ключ:**
    Приложение использует управление секретами Streamlit. для локальной разработки вы можете создать файл `.streamlit/secrets.toml` в корне проекта:
    ```toml
    # .streamlit/secrets.toml
    OPENAI_API_KEY = "your-openai-api-key"
    ```
    В качестве альтернативы, приложение предложит вам ввести ключ API в интерфейсе, если он не будет найден.

## Как запустить

Чтобы запустить приложение Streamlit, выполните следующую команду в корне проекта:

```bash
streamlit run advisor.py
```

Приложение будет доступно по адресу `http://localhost:8501`.
