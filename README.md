# ABACBS 2024 Hackathon - Agentic Workflow Project

## Table of Contents

- [Introduction](#introduction)
- [About the Hackathon](#about-the-hackathon)
- [Repository Structure](#repository-structure)
- [Background and Key Libraries](#background-and-key-libraries)
- [Installation](#installation)
  - [Prerequisites](#prerequisites)
  - [1. Install Poetry](#1-install-poetry)
  - [2. Clone the Repository and Install Dependencies](#2-clone-the-repository-and-install-dependencies)
  - [3. Test the Installation](#3-test-the-installation)
- [Using the Poetry Shell](#using-the-poetry-shell)
  - [Activating the Shell](#activating-the-shell)
  - [Deactivating the Shell](#deactivating-the-shell)
- [Installing and Using Ollama](#installing-and-using-ollama)
- [Downloading GGUF Models](#downloading-gguf-models)
- [Example Script: Snowball Chain for Blog Post Generation](#example-script-snowball-chain-for-blog-post-generation)
  - [Key Features](#key-features)
  - [Usage](#usage)
  - [Script Workflow](#script-workflow)
  - [Customization and Extension](#customization-and-extension)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This repository contains the Agentic Workflow project for the ABACBS 2024 Hackathon. The project demonstrates how to work with local Large Language Models (LLMs) in GGUF format using Python. It includes an example script that showcases a lightweight approach to constructing a prompt chain for generating blog posts.

## About the Hackathon

The ABACBS 2024 Hackathon is an event organized by the Australian Bioinformatics and Computational Biology Society. This project serves as a starting point for participants interested in exploring agentic workflows with local LLMs. The challenge which we set on the day was the learn how to use a local LLM to parse soft files from the Gene Expression Omnibus (GEO) and extract key pieces of information. 

## Repository Structure

- `src/`: Contains the main project code, including the example script `snowball_chain.py`.
- `models/`: Directory for storing downloaded GGUF model files.
- `snippets/`: Personal code snippets and examples (optional, not intended for hackathon use).

## Background and Key Libraries

This project leverages two main Python packages:

1. **llama-cpp-python**: A Python binding for the llama.cpp library, enabling efficient inference of LLaMA models locally. [GitHub Repository](https://github.com/abetlen/llama-cpp-python)
2. **instructor**: A library for structured outputs with LLMs, simplifying the handling of complex data structures. [GitHub Repository](https://github.com/jxnl/instructor)

These libraries allow for working with large language models on your local machine and extracting structured information from their outputs.

## Installation

### Prerequisites

- **Python 3.8 or higher**: Ensure you have Python installed. You can check your version with:
  ```
  python3 --version
  ```
- **Git**: To clone the repository.
- **System Dependencies**: Depending on your operating system, you may need to install additional libraries for llama-cpp-python. Refer to the [llama-cpp-python installation guide](https://github.com/abetlen/llama-cpp-python#installation) for details.

### 1. Install Poetry

Poetry is a dependency management and packaging tool for Python. Install Poetry by running:

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

After installation, ensure Poetry is added to your system's PATH. You may need to restart your terminal or follow any additional instructions provided by the installer.

For alternative installation methods, visit the [official Poetry documentation](https://python-poetry.org/docs/#installation).

### 2. Clone the Repository and Install Dependencies

Clone this repository and navigate to the project directory:

```bash
git clone https://github.com/TimoLassmann/ABACBS_2024_hackathon_agentic_wf.git
cd ABACBS_2024_hackathon_agentic_wf
```

Install the project dependencies using Poetry:

```bash
poetry install
```

This command creates a virtual environment and installs all necessary dependencies specified in the `pyproject.toml` file.

### 3. Test the Installation

Activate the virtual environment and run a test script:

```bash
poetry shell
python -c "import llama_cpp; import instructor; print('Installation successful!')"
```

If you see "Installation successful!", you're all set.

Troubleshooting:

- **Poetry Not Found**: Ensure that Poetry is added to your PATH.
- **Python Version**: Confirm you're using Python 3.8 or higher.
- **Compilation Errors**: If you encounter errors related to llama-cpp-python, install the required system libraries. Refer to the installation guide for assistance.

## Using the Poetry Shell

### Activating the Shell

Enter the virtual environment:

```bash
poetry shell
```

You'll see your shell prompt change to indicate you're now inside the virtual environment.

### Deactivating the Shell

To exit the virtual environment:

- Type `exit` and press Enter, or
- Press Ctrl+D.

Note: The `deactivate` command is not applicable when using Poetry's shell.

## Installing and Using Ollama

Ollama is a framework for running large language models locally, providing an alternative to using GGUF models directly.

### Install Ollama

For macOS and Linux:

```bash
curl https://ollama.ai/install.sh | bash
```

Note: Always review scripts before running them to ensure they're safe.

For more details, visit the [official Ollama documentation](https://ollama.ai/download).

### Start the Ollama Service

```bash
ollama serve
```

This command starts the Ollama server, which handles model inference requests.

### Pull a Model

Download a model compatible with Ollama:

- Mistral Model:
  ```bash
  ollama pull mistral
  ```

  Note: The example script is currently hard-coded to use the Mistral model.

- Llama 2 (9B):
  ```bash
  ollama pull llama3.2:3b
  ```

#### Hardware Requirements

- Models vary in size; ensure your system has enough RAM.
- The 7B and 9B models typically require at least 16GB of RAM.
- Larger models (e.g., 70B) require significantly more resources.

### Using Ollama with the Script

Run the `snowball_chain.py` script without specifying a model path to use Ollama:

```bash
poetry run python src/snowball_chain.py --topic "Your blog topic"
```

## Downloading GGUF Models

If you prefer to use local GGUF models with llama-cpp-python, follow these steps:

1. Select a GGUF Model:
   - [Llama 2 7B Chat (Q4_K_M)](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf)
   - [Mistral 7B Instruct (Q4_K_M)](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf)

   Note: Choose a model compatible with your system's RAM.

2. Place the Model in the `models/` Directory:
   ```bash
   mkdir -p models
   mv /path/to/downloaded/model.gguf models/
   ```

3. Run the Script with the Model Path:
   ```bash
   poetry run python src/snowball_chain.py --model_path models/model.gguf --topic "Your blog topic"
   ```

   Replace `model.gguf` with the actual filename of the model you downloaded.

## Example Script: Snowball Chain for Blog Post Generation

This project includes an example script (`src/snowball_chain.py`) demonstrating how to build a prompt chain using either a local LLM or an Ollama endpoint.

### Key Features

1. **Lightweight Prompt Chain**: Creates a "snowball effect" by feeding the output of one LLM call into the next.
2. **Structured Outputs with Instructor**: Uses the `instructor` library to capture outputs in structured Python classes.
3. **Flexible Model Integration**: Supports both local GGUF models and Ollama inference endpoints.

### Usage

With Ollama:
```bash
poetry run python src/snowball_chain.py --topic "Your blog topic"
```

With a Local GGUF Model:
```bash
poetry run python src/snowball_chain.py --model_path models/model.gguf --topic "Your blog topic"
```

### Script Workflow

1. Generate a Blog Post Title: Based on the provided topic.
2. Create a Hook: Develop a compelling introduction.
3. Generate Full Content: Includes sections like introduction, background, main content, and conclusion.

### Customization and Extension

Feel free to modify the script to suit your needs:

- **Adjust Prompts**: Customize the prompts to change how the LLM generates content.
- **Change Models**: Experiment with different models to see how outputs vary.
- **Extend Functionality**: Add new steps to the prompt chain for more complex workflows.

## Contributing

This project was created for the ABACBS 2024 Hackathon and is provided as-is for educational purposes. You're welcome to use, modify, and extend the code for your projects.

Please note:
- **Maintenance**: This repository is not actively maintained post-hackathon.
- **Issues**: You can open issues, but they may not be addressed promptly.

## License

[Insert your chosen license here]
