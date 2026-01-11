# Chat Archive Profile Extractor (AI-Powered)

This tool automatically extracts professional profiles from chat history (Telegram, Slack, WhatsApp) using OpenAI's GPT-4o-mini. It filters messages for introductions, parses structured data (Name, Role, Company, Contacts), and can optionally filter candidates by relevance to a specific job description.

## Features

-   **Multi-Platform Support**: Parses exports from Telegram (`result.json`), Slack (JSON), and WhatsApp (`_chat.txt`).
-   **AI extraction**: Uses LLM to intelligently extract structured data from unstructured text.
-   **Smart Filtering**:
    -   **Keyword Filter**: Pre-filters messages to reduce API costs.
    -   **Relevance Filter**: (Optional) Uses a second AI pass to filter candidates based on your specific query (e.g., "Python Developer with AI experience").
-   **Multilingual**: Supports English and Russian out of the box (configurable via `.env`).
-   **Cost Tracking**: detailed logs with token usage and estimated costs.
-   **Export**: Saves results to Excel (`.xlsx`) and CSV.

## Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/chat-profile-extractor.git
    cd chat-profile-extractor
    ```

2.  **Create a virtual environment**:
    ```bash
    python -m venv venv
    # Windows
    venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configuration**:
    Copy `.env.example` to `.env` and fill in your details:
    ```bash
    cp .env.example .env
    ```

    **Settings in `.env`**:
    *   `OPENAI_API_KEY`: Your OpenAI API Key.
    *   `LANGUAGE`: `EN` or `RU` (controls prompts and output language).
    *   `INPUT_FILE_TYPE`: `telegram`, `slack`, or `whatsapp`.
    *   `INPUT_FILENAME`: Path to your export file (e.g., `result.json`).
    *   `ENABLE_RELEVANCE_FILTER`: `True` or `False`.
    *   `RELEVANCE_QUERY`: Description of who you are looking for (e.g., "Senior Java Developer").

## Usage

1.  Place your chat export file in the project directory (or update `INPUT_FILENAME` in `.env`).
2.  Run the script:
    ```bash
    python main.py
    ```
3.  Check the `results/` folder:
    *   `all_candidates.xlsx`: All detected introductions.
    *   `relevant_candidates.xlsx`: Candidates matching your `RELEVANCE_QUERY`.
    *   `logs.txt`: Execution report with cost estimation.

## Project Structure

*   `main.py`: Entry point and logic.
*   `parsers.py`: Parsers for different chat formats.
*   `dictionaries/`: Keywords for pre-filtering (JSON).
*   `prompts/`: System and User prompts for the AI (Text files).
*   `i18n/`: Localization files for logs and headers.
*   `results/`: Output directory.

## License

MIT
