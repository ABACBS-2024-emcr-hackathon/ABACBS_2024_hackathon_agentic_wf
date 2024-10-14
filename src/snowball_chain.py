import instructor
import json
from llama_cpp import Llama
from instructor import patch
from pydantic import BaseModel, Field
from typing import List

from tqdm import tqdm
import time

import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description="Create a blog post based on a topic via a llm chain.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the GGUF model file")
parser.add_argument("--topic", type=str, default="Stars.", help="Input the topic of the blog post.")
args = parser.parse_args()

# Initialize the LLM
llm = Llama(
    model_path=args.model_path,
    verbose=False
)

patched_llm = instructor.patch(
    create=llm.create_chat_completion_openai_v1,
)

class blog_title(BaseModel):
    blog_title: str = Field(
            description="The title of the blog post.",
    )

class blog_hook(BaseModel):
    blog_hook: str = Field(
            description="The hook for the blog post.",
    )

class blog_paragraph(BaseModel):
    blog_paragraph: str = Field(
            description="The first paragraph of the blog post.",
    )

def get_blog_post_title(topic: str) -> blog_title:
    with tqdm(total=100, desc="Generating blog post title", unit="%") as pbar:
        response = patched_llm(
            response_model=blog_title,
            max_tokens=512,
            temperature=0.1,
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
        response = patched_llm(
            response_model=blog_hook,
            max_tokens=512,
            temperature=0.1,
            messages=[
              #  {"role": "system", "content": "You are an expert in generating blog hooks. You respond in JSON and always include the hook."},
                {"role": "user",
                    "content":f"""
                        Generate one hook for the blog post title: {title}
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

def get_blog_paragraph(title: str, hook: str) -> blog_paragraph:
    with tqdm(total=100, desc="Generating blog paragraph", unit="%") as pbar:
        response = patched_llm(
            response_model=blog_paragraph,
            response_format={"type": "json_object"},
            max_tokens=1024,
            temperature=0.1,
            messages=[
              # {"role": "system", "content": "You are an expert in writing interesting blog posts. You respond in JSON and always include several paragraphs worth of content."},
                {"role": "user", "content": f"""
                    Based on the TITLE:
                    {title}
                    and HOOK:
                    {hook}
                    generate the first paragraph of the blog post.
                    """},
            ],
        )

        # Simulate progress
        for _ in range(10):
            time.sleep(0.1)
            pbar.update(10)

    return response


if __name__ == "__main__":
    try:
        # Use command-line arguments
        title = get_blog_post_title(args.topic)
        print(f"Title: {title.blog_title}")
        hook = get_blog_hook(title.blog_title)
        print(f"Hook: {hook.blog_hook}")
        paragraph = get_blog_paragraph(title.blog_title, hook.blog_hook)
        print(f"Paragraph: {paragraph.blog_paragraph}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
