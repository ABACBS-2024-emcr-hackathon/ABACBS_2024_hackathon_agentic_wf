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
# --geo_soft: The geo_soft file

parser = argparse.ArgumentParser(description="Parse a GEO file  via a llm. Output JSON containg experimental meta data.")
parser.add_argument("--model_path", type=str, help="Path to the GGUF model file")
parser.add_argument("--geo_soft", type=str, help="Input the GEO soft file")
args = parser.parse_args()

# Initialize the counter for lines starting with "!Series_sample_id ="
sample_id_count = 0
geo_content = ""

# Open and parse the GEO soft file
if args.geo_soft:
    with open(args.geo_soft, 'r') as file:
        for line in file:
            # Add each line to geo_content to build the full file content
            geo_content += line
            
            # Increment the counter if line starts with "!Series_sample_id ="
            if line.startswith("!Series_sample_id ="):
                sample_id_count += 1

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

class geo_summary(BaseModel):
    number_of_samples: int = sample_id_count
    tissue: str = Field(description="The specific tissue type used in the study.")
    organism: str = Field(description="The organism from which the samples were taken.")
    molecule: str = Field(description="The type of molecule analyzed (e.g., DNA, RNA, protein).")
    genome_version: str = Field(description="The version or build of the genome used (e.g., hg38, mm10).")
    stranded: str = Field(description="Whether the data is stranded or unstranded (return: stranded or unstranded)")
    aligner: str = Field(description="The name of the aligner used (e.g., STAR, HISAT2).")


def parse_GEO_soft() -> geo_summary:
    geo_parser_prompt = f"""
    You are a language model tasked with analyzing a GEO soft file to extract key information about the study. The data you need to extract includes the following fields:

    - **Tissue**: The specific tissue type used in the study.
    - **Organism**: The organism (E.g. Homo sapiens, Mus musculus).
    - **Molecule**: The molecule (E.g. DNA or RNA).
    - **Genome Version**: Specifies the version or build of the genome used in the study (e.g., hg38 or T2T for human genome, mm10 for mouse genome etc.).
    - **Stranded**: If sequencing data is Stranded/Unstraned.
    - **Aligner**: Alignment program (e.g. bowtie2, STAR, hisat2 etc.)


    Here is the GEO soft file content:
    {geo_content}
    """
    response = model.generate(
        response_model=geo_summary,
        max_tokens=2048,
        temperature=0,
        max_retries=10,
        messages=[
            {"role": "user", "content": geo_parser_prompt}
        ],
    )
    return response

# Call the function and get the GeoSummary instance
geo_summary = parse_GEO_soft()

# Convert to JSON and format with indentation
formatted_json_output = json.dumps(json.loads(geo_summary.model_dump_json()), indent=4)

# Print the formatted JSON output
print(formatted_json_output)