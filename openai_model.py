from pydantic import BaseModel, Field
from typing import Any, Dict
from openai import OpenAI
import timeout_decorator
from dotenv import load_dotenv
import re
import time

load_dotenv()

def remove_markdown_syntax(text: str) -> str:
    # remove triple backtick code blocks (```python ... ```)
    text = re.sub(r"```[\s\S]*?```", lambda m: re.sub(r"^```.*\n|```$", '', m.group()), text)

    # remove inline code (`code`)
    text = re.sub(r"`([^`]*)`", r"\1", text)

    # remove bold (**text** or __text__)
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

    # remove italic (*text* or _text_)
    text = re.sub(r"\*(.*?)\*", r"\1", text)

    # remove blockquotes
    text = re.sub(r"^>\s?", '', text, flags=re.MULTILINE)

    text = text.replace("python", "")
    return text.strip()

def format_prompt(prompt: str, attr: dict, **kwargs) -> str:
    return prompt.format(**attr)

def add_metadata(total_metadata, metadata):
    total_metadata["input_tokens"] += metadata['input_tokens']
    total_metadata["output_tokens"] += metadata['output_tokens']
    total_metadata["text"] += "\n" + metadata["text"]

    if "content_used" in metadata.keys():
        total_metadata["content_used"] += metadata['content_used']
        total_metadata["total_content"] += metadata['total_content']
        total_metadata["num_tables"] += 1

    return total_metadata

def extract_result(text: str, pattern: str) -> str:
    position = text.lower().rfind(pattern.lower())
    if position == -1:
        print(f"Cannot find pattern '{pattern}' in '{text}'")
        return ""
    else:
        position += len(pattern)
    return text[position:].strip()

class OpenAIModel(BaseModel):
    model_name: str = Field("gpt-4o-mini", strict=True, description="Name of the openai model as per their official website")
    temperature: float = Field(.0000000000000000000001, strict=True, description="The temperature of the model in between 0 and 1")
    top_p: float = Field(1.0, strict=True, description="The top_p of the model in between 0 and 1")
    client: Any = None # OpenAI()
    max_retries: int = Field(50, strict=True, description="Number of retries in case of failed OpenAI API call")

    def init_client(self):
        self.client = OpenAI()

    #@timeout_decorator.timeout(60, timeout_exception=StopIteration)
    def call_gpt(self, prompt: str, delay: int = 2, backoff_factor: int = 2) -> (str, dict):
        for i in range(self.max_retries):
            try:
                completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {
                                "role": "user",
                                "content": f"{prompt}"
                            }
                        ],
                        temperature=self.temperature,
                        top_p=self.top_p,
                        seed=42,
                )
            except:
                if i == self.max_retries - 1:
                    raise

                time.sleep(delay)
                delay *= backoff_factor
                continue
            break

        metadata = {
            "input_tokens": completion.usage.prompt_tokens,
            "output_tokens": completion.usage.completion_tokens,
        }

        return completion.choices[0].message.content, metadata


    def query(self, prompt: str, attr: dict, **kwargs) -> tuple[str, dict]:
        if self.client is None:
            self.init_client()
        if len(attr) > 0:
            prompt = format_prompt(prompt, attr)
        text = prompt

        for _ in range(self.max_retries):
            try:
                response = self.call_gpt(prompt)
                text+="\n"+response[0]
                response[1]["text"] = text
                return response
            except StopIteration:
                print("Failed to get a response. Retrying...")

        raise RuntimeError(f"Failed to query OpenAI after {self.max_retries} retries.")
