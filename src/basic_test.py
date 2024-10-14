import instructor
from llama_cpp import Llama
from instructor import patch
from pydantic import BaseModel, Field
from typing import List

from tqdm import tqdm
import time

import argparse

# Set up the argument parser
parser = argparse.ArgumentParser(description="Get a movie recommendation using LLM.")
parser.add_argument("--model_path", type=str, required=True, help="Path to the GGUF model file")
parser.add_argument("--movie", type=str, default="A movie about bringing back dinosaurs.", help="Input a description of the movie you would like to see.")
args = parser.parse_args()

# Initialize the LLM
llm = Llama(
    model_path=args.model_path,
    verbose=False
)

patched_llm = instructor.patch(
    create=llm.create_chat_completion_openai_v1,
)

# Define the structured output using Pydantic
class MovieRecommendation(BaseModel):
    movietitle: str = Field(
            description="The title of the recommended movie.",
        )
    genre: str = Field(
            description="The genre of the recommended movie.",
        )
    year: int = Field(
            description="The year the movie was released.",
        )
    reasons: List[str] = Field(
            description="A list of reasons behind the recommendation.",
        )

# Function to get movie recommendations
def get_movie_recommendation(movie: str) -> MovieRecommendation:
    prompt = f"""Give me a movie recommendation based on this: {movie}"""

    with tqdm(total=100, desc="Generating recommendation", unit="%") as pbar:
        response = patched_llm(
            response_model=MovieRecommendation,
            max_tokens=512,
            temperature=0.7,
            messages=[
                {"role": "system", "content": "You are an expert in recommending movies to users. You respond in JSON and always include the title, genre, year, and reasons for your recommendation."},
                {"role": "user", "content": prompt},
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
        recommendation = get_movie_recommendation(args.movie)

        print(f"Title: {recommendation.movietitle}")
        print(f"Genre: {recommendation.genre}")
        print(f"Year: {recommendation.year}")
        print("Reasons:")
        for reason in recommendation.reasons:
            print(f"- {reason}")
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
