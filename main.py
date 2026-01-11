import asyncio
import json
import os
import re
import sys
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

import aiohttp
import pandas as pd
from dotenv import load_dotenv
from tqdm.asyncio import tqdm

from parsers import ChatMessage, TelegramParser, SlackParser, WhatsAppParser

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LANGUAGE = os.getenv("LANGUAGE", "EN").upper()  # RU or EN
INPUT_FILE_TYPE = os.getenv("INPUT_FILE_TYPE", "telegram").lower() # telegram, slack, whatsapp
ENABLE_RELEVANCE_FILTER = os.getenv("ENABLE_RELEVANCE_FILTER", "False").lower() == "true"
RELEVANCE_QUERY = os.getenv("RELEVANCE_QUERY", "")

# Output configuration
RESULTS_DIR = "results"
INPUT_FILE = "result.json" 
if os.getenv("INPUT_FILENAME"):
    INPUT_FILE = os.getenv("INPUT_FILENAME")

# Load internationalization (i18n)
def load_i18n(lang: str) -> Dict[str, Any]:
    filename = f"i18n/{lang.lower()}.json"
    if not os.path.exists(filename):
        logger.warning(f"Localization for language {lang} not found ({filename}). Using RU.")
        filename = "i18n/ru.json"
    
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

I18N = load_i18n(LANGUAGE)
LOGS = I18N.get("logs", {})
TABLE_HEADERS = I18N.get("table", {})

if not OPENAI_API_KEY:
    logger.warning(LOGS.get("api_key_warning"))

# Load dictionaries (keywords)
def load_dictionary(lang: str) -> Dict[str, Any]:
    filename = f"dictionaries/{lang.lower()}.json"
    if not os.path.exists(filename):
        logger.warning(LOGS.get("dictionary_warning", "").format(lang=lang, filename=filename))
        filename = "dictionaries/ru.json"
    
    with open(filename, 'r', encoding='utf-8') as f:
        return json.load(f)

DICTIONARY = load_dictionary(LANGUAGE)
KEYWORDS = DICTIONARY.get("keywords", {})

# Load prompts from files
def load_prompt_text(lang: str, filename: str) -> str:
    path = f"prompts/{lang.lower()}/{filename}"
    if not os.path.exists(path):
         # fallback to ru
         path = f"prompts/ru/{filename}"
    
    if not os.path.exists(path):
        return ""

    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()

SYSTEM_EXTRACTION_PROMPT = load_prompt_text(LANGUAGE, "system_extraction.txt")
USER_EXTRACTION_PROMPT = load_prompt_text(LANGUAGE, "user_extraction.txt")
SYSTEM_RELEVANCE_PROMPT = load_prompt_text(LANGUAGE, "system_relevance.txt")
USER_RELEVANCE_PROMPT = load_prompt_text(LANGUAGE, "user_relevance.txt")

@dataclass
class ProcessedProfile:
    original_message: ChatMessage
    is_introduction: bool
    data: Dict[str, Any]
    is_relevant: Optional[bool] = None
    relevance_reason: Optional[str] = None

def get_parser(file_type: str, input_file: str):
    if file_type == 'telegram':
        return TelegramParser(input_file)
    elif file_type == 'slack':
        return SlackParser(input_file)
    elif file_type == 'whatsapp':
        return WhatsAppParser(input_file)
    else:
        logger.error(LOGS.get("unknown_file_type", "").format(file_type=file_type))
        return TelegramParser(input_file)

def is_potential_intro(text: str) -> bool:
    """Check if the text matches the filtering criteria."""
    if not text or len(text) < 50:
        return False
        
    text_lower = text.lower()
    
    # Check for at least one match in any category
    for category, patterns in KEYWORDS.items():
        # patterns can be a list
        for pattern in patterns:
            if pattern.lower() in text_lower:
                return True
    return False

def filter_messages(messages: List[ChatMessage]) -> List[ChatMessage]:
    """Filter messages by keywords."""
    filtered = [
        msg for msg in messages 
        if is_potential_intro(msg.text)
    ]
    logger.info(LOGS.get("filtered_count", "").format(count=len(filtered)))
    return filtered

class OpenAIAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.semaphore = asyncio.Semaphore(10)  # Rate limiting
        self.base_url = "https://api.openai.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        # Usage stats
        self.total_requests = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.estimated_cost = 0.0
        
        # Free version often uses gpt-3.5 or now gpt-4o-mini
        # Cost for gpt-4o-mini: Input: $0.15/1M, Output $0.60/1M
        self.price_input_per_m = 0.15
        self.price_output_per_m = 0.60

    def _update_stats(self, usage: Dict):
        if not usage:
            return
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        
        self.total_requests += 1
        self.total_input_tokens += prompt_tokens
        self.total_output_tokens += completion_tokens
        
        cost = (prompt_tokens / 1_000_000 * self.price_input_per_m) + \
               (completion_tokens / 1_000_000 * self.price_output_per_m)
        self.estimated_cost += cost

    async def _call_gpt(self, session: aiohttp.ClientSession, system_prompt: str, user_prompt: str, message_id: str = "") -> Dict[str, Any]:
        payload = {
            "model": "gpt-4o-mini",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        retries = 3
        backoff = 1

        for attempt in range(retries):
            try:
                async with self.semaphore:
                    async with session.post(self.base_url, json=payload, headers=self.headers) as response:
                        if response.status == 200:
                            result = await response.json()
                            self._update_stats(result.get("usage", {}))
                            content = result['choices'][0]['message']['content']
                            return json.loads(content)
                        elif response.status == 429:
                            await asyncio.sleep(backoff)
                            backoff *= 2
                            continue
                        else:
                            logger.error(LOGS.get("api_error", "").format(status=response.status, text=await response.text()))
            except Exception as e:
                logger.error(LOGS.get("analysis_exception", "").format(e=e))
                if attempt == retries - 1:
                    break
                await asyncio.sleep(backoff)
                backoff *= 2
        
        return {}

    async def extract_profile(self, session: aiohttp.ClientSession, message: ChatMessage) -> ProcessedProfile:
        if not SYSTEM_EXTRACTION_PROMPT or not USER_EXTRACTION_PROMPT:
             logger.error("EXTRACTION PROMPTS not loaded.")
             return ProcessedProfile(message, False, {"error": "No prompts"})

        prompt = USER_EXTRACTION_PROMPT.replace("{text}", message.text)
        data = await self._call_gpt(session, SYSTEM_EXTRACTION_PROMPT, prompt, message.id)
        
        return ProcessedProfile(
            original_message=message,
            is_introduction=data.get('is_introduction', False),
            data=data
        )

    async def check_relevance(self, session: aiohttp.ClientSession, profile: ProcessedProfile) -> ProcessedProfile:
        if not profile.is_introduction or not ENABLE_RELEVANCE_FILTER:
            return profile
        
        if not SYSTEM_RELEVANCE_PROMPT or not USER_RELEVANCE_PROMPT:
             logger.error("RELEVANCE PROMPTS not loaded.")
             return profile 

        profile_json = json.dumps(profile.data, ensure_ascii=False)
        prompt = USER_RELEVANCE_PROMPT.replace("{profile_json}", profile_json).replace("{user_query}", RELEVANCE_QUERY)
        
        data = await self._call_gpt(session, SYSTEM_RELEVANCE_PROMPT, prompt, profile.original_message.id)
        
        profile.is_relevant = data.get('is_relevant', False)
        profile.relevance_reason = data.get('reason', '')
        return profile

    async def batch_process(self, messages: List[ChatMessage]) -> List[ProcessedProfile]:
        async with aiohttp.ClientSession() as session:
            # 1. Extraction
            tasks = [self.extract_profile(session, msg) for msg in messages]
            results = []
            for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc=LOGS.get("progress_bar", "Analysing messages")):
                result = await f
                results.append(result)
            
            # 2. Relevance Filter (if enabled)
            if ENABLE_RELEVANCE_FILTER:
                intro_candidates = [p for p in results if p.is_introduction]
                if intro_candidates:
                    relevance_tasks = [self.check_relevance(session, p) for p in intro_candidates]
                    for f in tqdm(asyncio.as_completed(relevance_tasks), total=len(relevance_tasks), desc=LOGS.get("progress_bar_relevance", "Filtering relevance")):
                        await f
            
            return results

def extract_username_from_text(text: str) -> Optional[str]:
    """Attempt to find @username in text."""
    match = re.search(r'(@[a-zA-Z0-9_]{5,})', text)
    if match:
        return match.group(1)
    return None

def extract_links_from_entities(entities: List[Dict]) -> List[str]:
    links = []
    if not entities: 
        return links
    for entity in entities:
        if entity.get('type') == 'text_link':
            links.append(entity.get('href'))
        elif entity.get('type') == 'url':
            pass
    return links

def save_results(profiles: List[ProcessedProfile], output_path: str):
    data_rows = []
    
    for p in profiles:
        msg = p.original_message
        d = p.data
        
        tg_username = d.get('telegram')
        if not tg_username:
            tg_username = extract_username_from_text(msg.text)
        
        links = extract_links_from_entities(msg.text_entities)
        other_contacts = d.get('other_contacts', '')
        if links:
            links_str = ", ".join(links)
            if other_contacts:
                other_contacts += LOGS.get("intro_link_text", " | Links: {links}").format(links=links_str)
            else:
                other_contacts = links_str
                
        expertise = d.get('expertise', [])
        if isinstance(expertise, list):
            expertise = ", ".join(expertise)

        row = {
            TABLE_HEADERS.get("name", "Name"): d.get('name'),
            TABLE_HEADERS.get("position", "Position"): d.get('position'),
            TABLE_HEADERS.get("company", "Company"): d.get('company'),
            TABLE_HEADERS.get("experience", "Experience"): d.get('experience_years'),
            TABLE_HEADERS.get("location", "Location"): d.get('location'),
            TABLE_HEADERS.get("expertise", "Expertise"): expertise,
            TABLE_HEADERS.get("looking_for", "Looking For"): d.get('looking_for'),
            TABLE_HEADERS.get("tg_name", "Telegram Name"): msg.from_name,
            TABLE_HEADERS.get("tg_id", "Telegram ID"): msg.from_id,
            TABLE_HEADERS.get("tg_username", "Telegram Username"): tg_username,
            TABLE_HEADERS.get("linkedin", "LinkedIn"): d.get('linkedin'),
            TABLE_HEADERS.get("other_contacts", "Other Contacts"): other_contacts,
            TABLE_HEADERS.get("summary", "Summary"): d.get('summary'),
            TABLE_HEADERS.get("original_text", "Original Text"): msg.text[:5000] if msg.text else "",
            TABLE_HEADERS.get("date", "Date"): msg.date
        }
        
        # Add relevance info only if it exists
        if p.is_relevant is not None:
             row[TABLE_HEADERS.get("relevance_reason", "Relevance Reason")] = p.relevance_reason or ""

        data_rows.append(row)

    if not data_rows:
        logger.warning(LOGS.get("no_intro_found"))
        return 0

    df = pd.DataFrame(data_rows)
    
    file_base = os.path.splitext(output_path)[0]
    output_xlsx = f"{file_base}.xlsx"
    output_csv = f"{file_base}.csv"

    # Save XLSX
    try:
        df.to_excel(output_xlsx, index=False)
        logger.info(LOGS.get("xlsx_saved", "").format(file=output_xlsx))
    except Exception as e:
        logger.error(LOGS.get("xlsx_error", "").format(e=e))

    # Save CSV
    try:
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        logger.info(LOGS.get("csv_saved", "").format(file=output_csv))
    except Exception as e:
        logger.error(LOGS.get("csv_error", "").format(e=e))
        
    return len(data_rows)

async def main():
    if not OPENAI_API_KEY:
        logger.error(LOGS.get("critical_api_key"))
        return
    
    # Ensure results directory exists
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    logger.info(LOGS.get("start_info", "").format(lang=LANGUAGE, type=INPUT_FILE_TYPE, file=INPUT_FILE, relevance=ENABLE_RELEVANCE_FILTER))

    parser = get_parser(INPUT_FILE_TYPE, INPUT_FILE)
    parser.load_messages()
    
    if not parser.messages:
        logger.info(LOGS.get("no_messages_processing"))
        return

    filtered_messages = filter_messages(parser.messages)
    
    if not filtered_messages:
        logger.info(LOGS.get("no_messages_filtered"))
        return

    analyzer = OpenAIAnalyzer(OPENAI_API_KEY)
    results = await analyzer.batch_process(filtered_messages)
    
    # Filter only successful introductions
    all_intros = [r for r in results if r.is_introduction]
    
    # 1. Save ALL found introductions
    path_all = os.path.join(RESULTS_DIR, "all_candidates")
    count_all = save_results(all_intros, path_all)
    
    # 2. Save only RELEVANT introductions (if filter enabled)
    relevant_intros = []
    if ENABLE_RELEVANCE_FILTER:
        relevant_intros = [r for r in all_intros if r.is_relevant]
        path_relevant = os.path.join(RESULTS_DIR, "relevant_candidates")
        count_relevant = save_results(relevant_intros, path_relevant)
    
    # 3. Create Logs file
    log_content = LOGS.get("summary_title", "Report") + "\n"
    log_content += "="*30 + "\n"
    log_content += f"Date: {datetime.now().isoformat()}\n"
    log_content += LOGS.get("summary_total", "").format(count=len(parser.messages)) + "\n"
    log_content += LOGS.get("summary_filtered", "").format(count=len(filtered_messages)) + "\n"
    log_content += LOGS.get("summary_found", "").format(count=len(all_intros)) + "\n"
    
    if ENABLE_RELEVANCE_FILTER:
        log_content += LOGS.get("summary_relevant", "").format(count=len(relevant_intros)) + "\n"
        
    log_content += LOGS.get("usage_stats", "").format(
        requests=analyzer.total_requests,
        input=analyzer.total_input_tokens,
        output=analyzer.total_output_tokens,
        cost=analyzer.estimated_cost
    )
    
    logs_path = os.path.join(RESULTS_DIR, "logs.txt")
    with open(logs_path, "w", encoding="utf-8") as f:
        f.write(log_content)

    # Console Summary
    print("\n" + "="*50)
    print(log_content)
    print(LOGS.get("summary_saved_all", "").format(file=path_all))
    if ENABLE_RELEVANCE_FILTER:
        print(LOGS.get("summary_saved_relevant", "").format(file=path_relevant))
    print("="*50 + "\n")

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
