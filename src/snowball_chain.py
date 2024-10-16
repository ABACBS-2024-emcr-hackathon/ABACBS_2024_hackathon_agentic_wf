# Import necessary libraries and modules
# instructor: For patching OpenAI's API with structured output support
# json: For JSON data handling
# OpenAI: The OpenAI API client
# Llama and LlamaPromptLookupDecoding: For local LLM support
# BaseModel and Field: For defining Pydantic models
# List: For type hinting
# tqdm: For progress bars
# time: For simulating progress
# argparse: For command-line argument parsing

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

# Parse command-line arguments:
# --model_path: Path to the GGUF model file (optional)
# --topic: The topic for the blog post (default is "Stars.")
# This allows users to specify a local model and customize the blog topic

parser = argparse.ArgumentParser(description="Create a blog post based on a topic via a llm chain.")
parser.add_argument("--model_path", type=str, help="Path to the GGUF model file")
parser.add_argument("--topic", type=str, default="Stars.", help="Input the topic of the blog post.")
args = parser.parse_args()

# ModelWrapper class:
# This class wraps the language model (either local Llama or Ollama) and provides a unified interface
# for generating responses. It handles the differences between local and Ollama models.
#
# Attributes:
#   model: The underlying language model (either local Llama or Ollama client)
#   is_ollama: Boolean flag indicating whether the model is Ollama (True) or local (False)
#
# Methods:
#   generate: Generates a response using the appropriate method based on the model type

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

# create_model function:
# This function initializes and returns a ModelWrapper object based on the provided model path.
# If a model path is provided, it sets up a local Llama model, otherwise it uses Ollama.
#
# Parameters:
#   model_path (str, optional): Path to the GGUF model file for local Llama
#
# Returns:
#   ModelWrapper: An instance of ModelWrapper containing either a local Llama model or Ollama client
#
# The function performs the following steps:
# 1. If a model_path is provided:
#    - Initialize a Llama model with specified parameters
#    - Patch the model with instructor for structured output support
#    - Wrap the patched model in a ModelWrapper (is_ollama=False)
# 2. If no model_path is provided:
#    - Initialize an OpenAI client configured for Ollama
#    - Patch the client with instructor for JSON output
#    - Wrap the patched client in a ModelWrapper (is_ollama=True)

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


# Pydantic models for structured output:
# These models define the expected structure of the language model's responses.
# They help enforce type checking and provide clear documentation of the data structure.

# blog_title: Represents the title of the blog post
# blog_hook: Represents the hook (introductory statement) of the blog post
# blog_content: Represents the main content of the blog post
#   - Contains a nested 'paragraphs' model for individual paragraphs
#   - Each paragraph has a 'message' (main point) and the actual 'paragraph' text


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



# Blog writer prompt:
# This multi-part prompt instructs the language model on how to generate a blog post.
# It includes the following sections:
# 1. Identity and purpose: Defines the model's role as an expert writer
# 2. Instructions: Provides guidelines for generating the blog post, including structure and content requirements
# 3. Title and Hook placeholders: To be filled with actual content during execution
# 4. Output Format: Specifies the expected structure of the response, including message and paragraph for each section
# The prompt aims to generate a well-structured, informative, and engaging blog post with 5-7 paragraphs.


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


# Blog post generation functions:
# These functions use the language model to generate different parts of the blog post.
# Each function utilizes a progress bar (tqdm) to provide visual feedback during generation.

# get_blog_post_title:
# Generates a title for the blog post based on the given topic.
# Parameters:
#   topic (str): The main topic of the blog post
# Returns:
#   blog_title: A Pydantic model containing the generated title

# get_blog_hook:
# Generates an engaging hook (introductory statement) for the blog post based on the title.
# Parameters:
#   title (str): The title of the blog post
# Returns:
#   blog_hook: A Pydantic model containing the generated hook

# get_blog_paragraph:
# Generates the main content of the blog post, including multiple paragraphs.
# Parameters:
#   title (str): The title of the blog post
#   hook (str): The hook of the blog post
# Returns:
#   blog_content: A Pydantic model containing the generated paragraphs, each with a message and content

# Each function follows a similar pattern:
# 1. Initialize a progress bar
# 2. Generate content using the language model
# 3. Simulate progress updates
# 4. Return the generated content



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

# Text wrapping function:
# This function wraps text to a specified width, maintaining indentation.
# It's used to format the output for better readability in the console.
#
# Parameters:
#   text (str): The input text to be wrapped
#   width (int): The maximum width of each line
#   indent (str): The indentation to be applied to each line
#
# Returns:
#   str: The wrapped and indented text
#
# The function performs the following steps:
# 1. Split the input text into paragraphs
# 2. For each paragraph, split into words and build lines up to the specified width
# 3. Apply indentation to each line
# 4. Join the lines and paragraphs back together

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

# Main execution block:
# This block is executed when the script is run directly (not imported as a module).
# It orchestrates the blog post generation process using the functions defined above.

# The process follows these steps:
# 1. Generate a blog post title based on the provided topic
# 2. Generate a hook based on the title
# 3. Generate the main content (body) of the blog post
# 4. Print the generated content in a formatted manner

# Error handling:
# - The entire process is wrapped in a try-except block to catch KeyboardInterrupt exceptions,
#   allowing the user to cancel the operation gracefully.

# Output formatting:
# - The title and hook are printed first, followed by each section of the body
# - Text wrapping is applied to ensure proper formatting in the console output
# - Each section is numbered and includes both the paragraph text and its main message
# - Separator lines are printed between sections for better readability

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
