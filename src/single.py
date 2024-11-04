import instructor
import json
from openai import OpenAI
from llama_cpp import Llama
from llama_cpp.llama_speculative import LlamaPromptLookupDecoding
from instructor import patch
from pydantic import BaseModel, Field
from typing import List
from tqdm import tqdm
import time
import argparse

parser = argparse.ArgumentParser(description="Create a blog post based on a topic via a llm chain.")
parser.add_argument("--model_path", type=str, help="Path to the GGUF model file")
parser.add_argument("--prompt", type=str, help="Input the topic of the blog post.")
args = parser.parse_args()

class ModelWrapper:
    def __init__(self, model, is_ollama: bool):
        self.model = model
        self.is_ollama = is_ollama

    def generate(self, response_model, messages, **kwargs):
        if self.is_ollama:
            return self.model.chat.completions.create(
                model="mistral:latest",
                response_model=response_model,
                messages=messages,
                **kwargs
            )
        else:
            return self.model(
                response_model=response_model,
                messages=messages,
                **kwargs
            )

def create_model(model_path: str = None) -> ModelWrapper:
    if model_path:
        llm = Llama(
            model_path=model_path,
            n_gpu_layers=-1,
            n_batch=512,
            chat_format="chatml",
            cache_prompt=False,
            n_ctx=2048,
            draft_model=LlamaPromptLookupDecoding(num_pred_tokens=10),
            logits_all=True,
            verbose=False,
        )
        patched_llm = instructor.patch(
            create=llm.create_chat_completion_openai_v1,
            mode=instructor.Mode.JSON_SCHEMA,
        )
        return ModelWrapper(patched_llm, is_ollama=False)
    else:
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="secret12345",
        )
        patched_client = instructor.patch(client, mode=instructor.Mode.JSON)
        return ModelWrapper(patched_client, is_ollama=True)


model = create_model(args.model_path)

class llm_respose(BaseModel):
    response: str = Field(description="The response to the users question")
    reasoning: str = Field(description="The reasoning behind the title.")


def get_rsponse(topic: str) -> llm_respose:
    with tqdm(total=100, desc="Generating response", unit="%") as pbar:
        response = model.generate(
            response_model=llm_respose,
            max_retries=10,
            temperature=0.7,
            messages=[
                {"role": "user", "content": f"""
                    Respons to the user query: {topic}.
                    Your response should be in the format of a JSON object. """},
            ],
        )

        # Simulate progress
        for _ in range(10):
            time.sleep(0.1)
            pbar.update(10)

    return response

def wrap_text(text, width, indent):
    lines = []
    for paragraph in text.split('\n'):
        words = paragraph.split()
        current_line = indent
        for word in words:
            if len(current_line) + len(word) + 1 <= width + len(indent):
                current_line += " " + word if current_line != indent else word
            else:
                lines.append(current_line)
                current_line = indent + word
        if current_line:
            lines.append(current_line)
    return '\n'.join(lines)

if __name__ == "__main__":
    try:
        # Use command-line arguments
        response = get_rsponse(args.prompt)

        print(f"Response:\n{wrap_text(response.response, 78, ' '*22)}")
        print(f"Reasoning:\n{wrap_text(response.reasoning, 78, ' '*22)}")

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
