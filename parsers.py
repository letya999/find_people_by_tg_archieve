import json
import re
import os
import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from abc import ABC, abstractmethod
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ChatMessage:
    id: str  # Changed to str to accommodate different ID types
    date: str
    from_name: str
    from_id: str
    text: str
    text_entities: List[Dict] = None
    username: Optional[str] = None

class BaseParser(ABC):
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.messages: List[ChatMessage] = []

    @abstractmethod
    def load_messages(self):
        """Load messages from the input file."""
        pass

class TelegramParser(BaseParser):
    def load_messages(self):
        """Parsing Telegram JSON export."""
        if not os.path.exists(self.input_file):
            logger.error(f"File {self.input_file} not found.")
            return
            
        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            raw_messages = data.get('messages', [])
            logger.info(f"Telegram: Loaded {len(raw_messages)} raw messages.")
            
            for msg in raw_messages:
                if msg.get('type') != 'message':
                    continue
                
                text_content = msg.get('text', '')
                full_text = ""
                if isinstance(text_content, list):
                    for item in text_content:
                        if isinstance(item, str):
                            full_text += item
                        elif isinstance(item, dict):
                            text_val = item.get('text', '')
                            if item.get('type') == 'text_link':
                                href = item.get('href', '')
                                if href:
                                    full_text += f"[{text_val}]({href})"
                                else:
                                    full_text += text_val
                            else:
                                full_text += text_val
                else:
                    full_text = str(text_content)

                # Parsing date if needed, keeping as string for now
                
                chat_msg = ChatMessage(
                    id=str(msg.get('id')),
                    date=msg.get('date'),
                    from_name=msg.get('from', 'Unknown'),
                    from_id=str(msg.get('from_id', '')),
                    text=full_text,
                    text_entities=msg.get('text_entities', [])
                )
                self.messages.append(chat_msg)
                
            logger.info(f"Telegram: Successfully parsed {len(self.messages)} messages.")
            
        except Exception as e:
            logger.error(f"Error reading Telegram file: {e}")

class SlackParser(BaseParser):
    def load_messages(self):
        """Parsing Slack JSON export (an array of message objects)."""
        if not os.path.exists(self.input_file):
            logger.error(f"File {self.input_file} not found.")
            return

        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # If data is a list (message history), process it. 
            # If it's a dict (wrapper), look for messages key.
            if isinstance(data, dict):
                 raw_messages = data.get('messages', [])
            elif isinstance(data, list):
                 raw_messages = data
            else:
                 logger.error("Slack: Unknown JSON structure.")
                 return

            logger.info(f"Slack: Loaded {len(raw_messages)} raw messages.")

            for msg in raw_messages:
                if msg.get('type') != 'message' or msg.get('subtype'):
                     # Skip system messages/subtypes often
                     pass

                text = msg.get('text', '')
                user_profile = msg.get('user_profile', {})
                from_name = user_profile.get('real_name') or user_profile.get('display_name') or msg.get('user', 'Unknown')
                
                ts = msg.get('ts')
                # Convert ts to date str
                try:
                    dt_object = datetime.fromtimestamp(float(ts))
                    date_str = dt_object.isoformat()
                except:
                    date_str = str(ts)

                chat_msg = ChatMessage(
                    id=str(msg.get('client_msg_id') or ts),
                    date=date_str,
                    from_name=from_name,
                    from_id=msg.get('user', ''),
                    text=text,
                    text_entities=[] # Slack logic for links could be added here
                )
                self.messages.append(chat_msg)

            logger.info(f"Slack: Successfully parsed {len(self.messages)} messages.")

        except Exception as e:
            logger.error(f"Error reading Slack file: {e}")

class WhatsAppParser(BaseParser):
    def load_messages(self):
        """Parsing WhatsApp _chat.txt export."""
        if not os.path.exists(self.input_file):
            logger.error(f"File {self.input_file} not found.")
            return

        try:
            with open(self.input_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            logger.info(f"WhatsApp: Loaded {len(lines)} lines.")
            
            # Pattern for: [21.01.21, 12:00:00] Author: Message
            # Simplified regex to catch most common formats
            # Attempt 1: Brackets date
            pattern = re.compile(r'^\[?(\d{1,4}[./-]\d{1,2}[./-]\d{1,4}[,\s]+\d{1,2}:\d{2}(?::\d{2})?)\]?[\s-]*([^:]+): (.*)$')
            
            for index, line in enumerate(lines):
                line = line.strip()
                if not line:
                    continue
                
                match = pattern.match(line)
                if match:
                    date_str, author, text = match.groups()
                    if "Messages and calls are end-to-end encrypted" in line:
                         continue

                    chat_msg = ChatMessage(
                        id=str(index),
                        date=date_str,
                        from_name=author,
                        from_id=author, # WA doesn't expose ID in txt
                        text=text,
                        text_entities=[]
                    )
                    self.messages.append(chat_msg)
                else:
                    # Multiline message continuation?
                    if self.messages:
                        self.messages[-1].text += f"\n{line}"

            logger.info(f"WhatsApp: Successfully parsed {len(self.messages)} messages.")

        except Exception as e:
            logger.error(f"Error reading WhatsApp file: {e}")
