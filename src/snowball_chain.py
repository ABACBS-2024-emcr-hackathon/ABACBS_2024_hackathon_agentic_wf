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
                model="gemma2:9b",
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
            n_ctx=16384,
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
    blog_introduction: str = Field(
            description="The introduction of the blog post.",
    )
    blog_background: str = Field(
            description="A section providing background to the topic discussed in the blog post.",
    )
    blog_mainsection: str = Field(
            description="The main body og the blog post.",
    )
    blog_conclusion: str = Field(
            description="The conclusion.",
    )

blog_writer_prompt = """
# Identity and purpose

You are a highly skilled expert writer tasked with generating a blog post in a structured JSON format.
The blog post should be divided into the following sections: `blog_introduction`, `blog_background`, `blog_mainsection`, and `blog_conclusion`.

# Instructions:

- You will generate a blog post based on the following **TITLE**: "{title}" and **HOOK**: "{hook}".
- The blog post should be informative, well-structured, and engaging. Make sure to use smooth transitions between sections.
- Each section should follow these guidelines:

  - **blog_introduction**: Provide a brief, compelling introduction to the topic. Engage the reader with a hook and explain why the topic is relevant or interesting. The introduction should be between 100-150 words and clearly set the stage for the topic.
  - **blog_background**: Offer detailed background information. Include historical details or context that helps the reader understand the topic's significance. This section should be between 150-200 words and should thoroughly explain any relevant developments or key points leading up to the main discussion.
  - **blog_mainsection**: Dive deeply into the topic, presenting key points, insights, and analysis. This section should elaborate on the topic comprehensively, addressing various perspectives and providing thoughtful commentary. It should be the longest section, around 300-400 words, and cover the key ideas thoroughly.
  - **blog_conclusion**: Summarize the main points of the blog post, offer final thoughts, and potentially suggest future directions or a call to action. The conclusion should be around 100-150 words and leave the reader with a clear takeaway or final thought on the subject.

### Output Format:
- Respond in JSON format with the following fields:
  - blog_introduction
  - blog_background
  - blog_mainsection
  - blog_conclusion

### Example Output:

{{{{
  "blog_introduction": "In recent years, AI has transformed various industries. One of the most notable areas where AI has made a significant impact is healthcare. In this post, we explore how AI is revolutionizing medical diagnostics.",
  "blog_background": "AI in healthcare is not a new concept. From the early days of expert systems to today's deep learning models, the application of AI has continuously evolved. However, recent advancements have made AI more accessible and powerful, allowing for more precise diagnoses and treatment plans.",
  "blog_mainsection": "One of the key breakthroughs in AI for healthcare is its ability to process vast amounts of data quickly. For example, AI algorithms can analyze medical images, detect anomalies, and assist in diagnosing conditions that may be difficult for human doctors to identify. Furthermore, AI-powered tools can assist in predictive analytics, helping healthcare professionals foresee potential complications before they arise.",
  "blog_conclusion": "In conclusion, AI is poised to continue reshaping the healthcare landscape. Its ability to improve diagnostics, personalize treatment plans, and analyze complex data is transforming how doctors care for patients. As we move forward, the integration of AI in healthcare will only deepen, promising better outcomes and more efficient medical practices."
}}}}

Ensure the content is clearly written, informative, and maintains a consistent tone throughout.
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

if __name__ == "__main__":
    try:
        # Use command-line arguments
        title = get_blog_post_title(args.topic)
        print(f"Title: {title.blog_title}")
        hook = get_blog_hook(title.blog_title)
        print(f"Hook: {hook.blog_hook}")
        paragraph = get_blog_paragraph(title.blog_title, hook.blog_hook)
        print(f"Intro: {paragraph.blog_introduction}")
        print(f"Background: {paragraph.blog_background}")
        print(f"Main Section: {paragraph.blog_mainsection}")
        print(f"Conclusion: {paragraph.blog_conclusion}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
