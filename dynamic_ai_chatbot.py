import os
import tiktoken
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

DEFAULT_API_KEY = os.environ.get('TOGETHER_AI_API_KEY')
DEFAULT_BASE_URL = 'https://api.together.xyz/v1'
DEFAULT_MODEL = 'meta-llama/Llama-3-8b-chat-hf'
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 512
DEFAULT_TOKEN_BUDGET = 4096
DEFAULT_SYSTEM_MESSAGE = "You are a sassy assistant who is fed up with answering questions."
DEFAULT_HISTORY_FILE = f'conversation_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'


class ConversationManager:

    def __init__(self,
                 api_key = None,
                 base_url = None,
                 model = None,
                 temperature = None,
                 max_tokens = None,
                 token_budget = None,
                 system_message = None,
                 history_file = None,
                 ):
        
        self.api_key = api_key or DEFAULT_API_KEY
        self.base_url = base_url or DEFAULT_BASE_URL
        self.client = OpenAI(api_key = self.api_key, base_url = self.base_url)
        self.model = model or DEFAULT_MODEL
        self.temperature = temperature or DEFAULT_TEMPERATURE
        self.max_tokens = max_tokens or DEFAULT_MAX_TOKENS
        self.token_budget = token_budget or DEFAULT_TOKEN_BUDGET
        self.system_message = system_message or DEFAULT_SYSTEM_MESSAGE
        self.history_file = history_file or DEFAULT_HISTORY_FILE
        self. conversation_history = [
                     {"role": "system", "content": self.system_message},
                 ]
        self.system_messages = {
            "sassy_assistant": "You are a sassy assistant who is fed up with answering questions.",
            "angry_assistant": "You are an angry assistant that likes yelling in all caps.",
            "thoughtful_assistant": "You are a thoughtful assistant, always ready to dig deeper to ensure understanding.",
            "custom": "Enter your custom system message here."
        }
        self.system_message = self.system_messages["thoughtful_assistant"]
        self.load_conversation_history()

    def set_persona(self, persona):
        try:
            self.system_message = self.system_messages[persona]
            self.update_system_message_in_history()
        except ValueError:
            print(f'{persona} is not found!')
            print(f'Available personas: {list(self.system_messages.keys())}')
            
    def set_custom_system_message(self, custom_message):
        self.system_messages["custom"] = custom_message
        return self.set_persona("custom")
    
    def update_system_message_in_history(self):
        if self.conversation_history and self.conversation_history[0]["role"] == "system":
            self.conversation_history[0]["content"] = self.system_message
        else:
            self.conversation_history.insert(0, {"role": "system", "content": self.system_message})

    def load_conversation_history(self):
        '''Reads conservation history from disk'''
        try:
            with open(self.history_file, 'r') as file:
                self.conversation_history = json.load(file)
        except FileNotFoundError:
            self.conversation_history = [{"role": "system", "content": self.system_message}]
        except json.JSONDecodeError:
            print('Error reading the conversation history file. Starting with an initial history.')
            self.conversation_history = [{"role": "system", "content": self.system_message}]
            
    def save_conversation_history(self):
        '''Writes conversation history to disk'''
        try:
            with open(self.history_file, "w") as file:
                json.dump(self.conversation_history, file, indent = 4)
        except IOError as e:
            print(f"An unexpected error occurred while saving file: {e}")
        except Exception as e:
            print(f"An unexpected error occurred while writing to file: {e}")
        
    def count_tokens(self, text):
        '''Returns the number of tokens in text'''

        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding('cl100k_base')
        tokens = encoding.encode(text)
        return len(tokens)
    
    @property
    def total_tokens_used(self):
        '''Return the total number of tokens used in the conversation history'''

        return sum(self.count_tokens(message['content']) for message in self.conversation_history)
    
    def enforce_token_budget(self):
        '''Removes messages from history to enforce token limit'''
        try:
            while self.total_tokens_used > self.token_budget:
                if len(self.conversation_history) <= 1:
                    break
                self.conversation_history.pop(1)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

    def chat_completion(self, prompt, temperature = None, max_tokens = None):
        '''Sends prompts to open api and returns responses'''

        self.conversation_history.append({"role": "user", "content": prompt})
        messages = [
        {"role": "system", "content": self.system_message},
        {"role": "user", "content": prompt}
        ]

        self.enforce_token_budget()
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                max_tokens = max_tokens or self.max_tokens,
                messages = messages,
                temperature = temperature or self.temperature,
            )
        except Exception as e:
            print(f"An unexpected error occurred during API call: {e}")
            return None

        ai_response = response.choices[0].message.content
        self.conversation_history.append({"role": "assistant", "content": ai_response})
        self.enforce_token_budget()
        self.save_conversation_history()

        return ai_response
    
