# Personal Code Snippets

This directory contains various code snippets and examples that are not part of the main project but may be useful for future reference or development. These snippets are not intended for use by hackathon participants.

## Contents

### 1. [Instructor Snippets](#instructor-snippets)

Nice way to structure user data using LLMs into a class and assign a method to use the data. This is useful for handling responses from LLMs in a more structured way.

``` python
from pydantic import BaseModel

class SearchQuery(BaseModel):
    query: str
    query_type: str

    def execute(self):
        print(f"Searching for {self.query} of type {self.query_type}")
        return "Results for " + self.query

client = instructor.from_openai(OpenAI())
response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Search for a picture of a cat"}],
    response_model=SearchQuery,
)

results = response.execute()  # Here, execute is called on the response object
```


## Usage

These snippets are for personal reference only. They may not be fully tested or integrated with the main project.

## Contributing

As these snippets are for personal use, contributions are not expected. However, if you have suggestions or improvements, feel free to create an issue in the main repository.

## License

These snippets are subject to the same license as the main project. Refer to the LICENSE file in the root directory for more information.
