# ABACBS 2024 Hackathon Agentic Workflow

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Downloading GGUF Models](#downloading-gguf-models)
- [Background and Key Libraries](#background-and-key-libraries)
- [Using the Poetry Shell](#using-the-poetry-shell)
- [Example Script: Snowball Chain](#example-script-snowball-chain-for-blog-post-generation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository contains the Agentic Workflow project for the ABACBS 2024 Hackathon. It utilizes Python packages such as llama-cpp-python and instructor to work with local large language models (LLMs) in GGUF format.

### About the Hackathon

The ABACBS 2024 Hackathon is an event organized by the Australian Bioinformatics and Computational Biology Society. This project serves as a starting point for participants interested in exploring agentic workflows with local LLMs. For more information about the hackathon, please visit the ABACBS website.

## Repository Structure

- `src/`: Contains the main project code, including the example script.
- `models/`: Directory for storing downloaded GGUF model files.
- `snippets/`: Personal code snippets and examples (not for hackathon use).


## Installation

### 1. Install Poetry

Poetry is a dependency management and packaging tool for Python. To install Poetry, run the following command in your terminal:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

For more detailed instructions or alternative installation methods, visit the [official Poetry documentation](https://python-poetry.org/docs/#installation).

### 2. Install Project Dependencies

Clone this repository and navigate to the project directory. Then run:

```bash
poetry install
```

This command will create a virtual environment and install all the necessary dependencies specified in the `pyproject.toml` file.

### 3. Test the Installation

To ensure everything is set up correctly, activate the virtual environment and run a test script:

```bash
poetry shell
python -c "import llama_cpp; import instructor; print('Installation successful!')"
```

If you encounter any issues:
- Make sure you have the latest version of Poetry installed
- Check that your Python version is compatible (this project requires Python 3.8+)
- If you face any compilation errors related to llama-cpp-python, you might need to install additional system libraries. Refer to the [llama-cpp-python documentation](https://github.com/abetlen/llama-cpp-python) for platform-specific instructions.

## Downloading GGUF Models

To use this project, you'll need to download a GGUF-formatted language model. Here are some options:

1. [Llama-3.2-3B-Instruct-Q8_0.gguf](https://huggingface.co/lmstudio-community/Llama-3.2-3B-Instruct-GGUF/resolve/main/Llama-3.2-3B-Instruct-Q8_0.gguf?download=true)
2. [gemma-2-9b-it-Q4_K_M.gguf](https://huggingface.co/lmstudio-community/gemma-2-9b-it-GGUF/resolve/main/gemma-2-9b-it-Q4_K_M.gguf?download=true)

Download your chosen model and place it in the `models` directory of this project.

## Background and Key Libraries

This project leverages two main Python packages:

1. **llama-cpp-python**: A Python binding for the llama.cpp library, which allows for efficient inference of LLaMA models. [GitHub Repository](https://github.com/abetlen/llama-cpp-python)

2. **instructor**: A library for structured outputs with LLMs, making it easier to work with complex data structures. [GitHub Repository](https://github.com/jxnl/instructor)

These libraries enable us to work with large language models locally and extract structured information from their outputs.

## Using the Poetry Shell

### Activating the Shell

To activate the Poetry shell and enter the virtual environment, use:

```bash
poetry shell
```

This command activates the virtual environment associated with your project.

### Deactivating the Shell

To exit (deactivate) the Poetry shell and return to your normal shell, you can use any of these methods:

1. Type `exit` and press Enter
2. Use the keyboard shortcut Ctrl+D
3. Type `deactivate` and press Enter

Any of these methods will exit the Poetry shell and return you to your system's default shell.

Remember, deactivating the shell doesn't delete or modify your virtual environment; it simply exits it. You can always reactivate it later using `poetry shell`.

## Example Script: Snowball Chain for Blog Post Generation

This repository includes an example script (`src/snowball_chain.py`) that demonstrates a lightweight approach to constructing a prompt chain using a local LLM. This script serves as a basic demonstration of how to work with local LLMs and can be used as a starting point for more complex applications.

The snowball prompt concept in this script was inspired by [this gist](https://gist.github.com/disler/d51d7e37c3e5f8d277d8e0a71f4a1d2e) by [disler](https://gist.github.com/disler).

### Key Features

1. **Lightweight Prompt Chain**: The script showcases an extremely lightweight approach to constructing a prompt chain, where the output of one LLM call is used as input for the next call. This creates a "snowball effect" in generating content.

2. **Structured Outputs with Instructor**: We use the `instructor` library to ensure the LLM's output is captured in Python classes. This approach allows for easy handling and passing of outputs within the code, maintaining type safety and structure.

3. **Local LLM Integration**: The script demonstrates how to work with local LLMs using the `llama-cpp-python` library, providing a foundation for participants to build upon or use as inspiration for their own implementations.

### Usage

To run the example script:

```bash
poetry run python src/snowball_chain.py --model_path /path/to/your/gguf/model --topic "Your blog topic"
```

Replace `/path/to/your/gguf/model` with the actual path to your downloaded GGUF model file, and "Your blog topic" with the desired topic for the blog post.

### Script Workflow

1. The script first generates a blog post title based on the given topic.
2. It then creates a hook for the blog post using the generated title.
3. Finally, it generates the full content of the blog post, including introduction, background, main section, and conclusion.

### Customization and Extension

While this script provides a basic demonstration, hackathon participants are encouraged to use this as a starting point and freely modify or extend it to accomplish their specific goals. The modular structure allows for easy additions or alterations to the prompt chain, output structures, or overall workflow.

Remember, this is just one approach to working with local LLMs. Feel free to explore alternative methods or libraries that best suit your project's needs.

## Contributing

This project was created for the ABACBS 2024 Hackathon and is provided as-is for demonstration and learning purposes. While there are no formal contribution guidelines, you are welcome to use, modify, and extend the code as you see fit for your own projects or during the hackathon.

Please note:
- This repository is not actively maintained.
- There is no expectation of ongoing development or updates after the hackathon.
- Feel free to fork the repository and adapt it to your needs.
- If you discover any critical issues or have suggestions, you can open an issue, but please be aware that responses may be limited or not forthcoming.

The primary goal is to provide a starting point and inspiration for working with local LLMs. We encourage you to experiment, learn, and build upon this code in whatever way serves your projects best.

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
