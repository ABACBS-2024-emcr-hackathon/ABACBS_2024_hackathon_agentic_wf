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

# Set up the argument parser
parser = argparse.ArgumentParser(description="Create a blog post based on a topic via a llm chain.")
parser.add_argument("--model_path", type=str, help="Path to the GGUF model file")
parser.add_argument("--topic", type=str, default="Stars.", help="Input the topic of the blog post.")
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
            verbose=True,
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


class blog_title(BaseModel):
    blog_title: str = Field(
            description="The title of the blog post.",
    )

class blog_hook(BaseModel):
    blog_hook: str = Field(
            description="The hook for the blog post.",
    )

class blog_content(BaseModel):
    class paragraphs(BaseModel):
        message: str = Field(description="The main message the paragraph should communicate.")
        paragraph: str = Field(description="A three to five sentence long paragraph.")
    content: list[paragraphs]

blog_writer_prompt = """
# Identity and purpose

You are a highly skilled expert writer tasked with generating a blog post in a structured JSON format.

# Instructions:

- You will generate a blog post based on the following **TITLE** and **HOOK**.
- The blog post should be informative, well-structured, and engaging. Make sure to use smooth transitions between paragraphs.
- Each paragraph should clearly communicate one main message and be three to five sentences long.
- Aim for a total of 5-7 paragraphs in the blog post.
- Try not to repeat content from the HOOK in the paragraphs. Instead, expand on the ideas presented in the HOOK.

## TITLE
{title}

## HOOK
{hook}
## Output Format:
- Respond with a list of paragraphs, each containing:
  - "message": A brief summary of the main point or message of that paragraph
  - "paragraph": The full text of the paragraph (3-5 sentences)

Ensure the content is clearly written, informative, and maintains a consistent tone throughout. Aim for 5-7 paragraphs in total.
"""


def get_blog_post_title(topic: str) -> blog_title:
    with tqdm(total=100, desc="Generating blog post title", unit="%") as pbar:
        response = model.generate(
            response_model=blog_title,
            max_retries=10,
            temperature=0.7,
            messages=[
                {"role": "user", "content": f"""
                    Generate one blog post title about: {topic}.
                    Your response should be in the format of a JSON object. """},
            ],
        )

        # Simulate progress
        for _ in range(10):
            time.sleep(0.1)
            pbar.update(10)

    return response

def get_blog_hook(title: str) -> blog_hook:
    with tqdm(total=100, desc="Generating blog hook", unit="%") as pbar:
        response = model.generate(
            response_model=blog_hook,
            max_tokens=512,
            temperature=0.7,
            max_retries=10,
            messages=[
                {"role": "user",
                    "content":f"""
                        Create an engaging hook for a blog post titled ‘{title}.’
                        The hook should be a thought-provoking statement or question that draws readers in,
                        highlighting the significance of precision medicine and its impact on healthcare. Aim
                        for a tone that is informative yet captivating.
                        Your response should be in the format of a JSON object.
                        """
                },
            ],
        )

        # Simulate progress
        for _ in range(10):
            time.sleep(0.1)
            pbar.update(10)

    return response

def get_blog_paragraph(title: str, hook: str) -> blog_content:
    with tqdm(total=100, desc="Generating blog paragraph", unit="%") as pbar:

        formatted_prompt = blog_writer_prompt.format(title=title, hook=hook)
        response = model.generate(
            response_model=blog_content,
            max_tokens=2048,
            temperature=0.7,
            max_retries=10,
            messages=[
                {"role": "user", "content": formatted_prompt},
            ],
        )

        # Simulate progress
        for _ in range(10):
            time.sleep(0.1)
            pbar.update(10)

    return response


# Print the blog content with aligned labels
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
        title = get_blog_post_title(args.topic)

        hook = get_blog_hook(title.blog_title)

        body = get_blog_paragraph(title.blog_title, hook.blog_hook)

        print(f"Title:\n{wrap_text(title.blog_title, 78, ' '*22)}")
        print(f"Hook:\n{wrap_text(hook.blog_hook, 78, ' '*22)}")
        for i, section in enumerate(body.content, 1):
            print(f"\nSection #{i}:")
            print(f"Paragraph:\n{wrap_text(section.paragraph, 78, ' '*22)}")
            print(f"Message:\n{wrap_text(section.message, 78 , ' '*22)}")
            print("-" * 50)  # Print a separator line

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
