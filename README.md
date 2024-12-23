# Ollama Model Inference Speed Measurement

This repository contains a Python script to measure the inference speed of 
[Ollama](https://ollama.com/) models based on specified prompts. The script evaluates the 
performance in terms of tokens per second (TPS) for both prompt evaluation 
and response generation.

## Prerequisites
- Python 3.8 or higher
- Access to the Ollama library and models

## Installation

### Clone the Repository
First, clone this repository to your local machine using `git`:

```bash
git clone https://github.com/dearliory/ollama-inference-speed.git
cd ollama-inference-speed
```
or

```bash
git clone git@github.com:dearliory/ollama-inference-speed.git
cd ollama-inference-speed
```

### Install Dependencies
The script requires the `ollama` and `pandas` packages. These are listed 
in the `requirements.txt` file. You can install them using `pip`:

```bash
python -m pip install -r requirements.txt
```

## Usage

### Basic Usage
To run the script with default settings, use the following command:

```bash
python inference_speed.py --verbose
```

This will evaluate the model `llama3.1:latest` with the prompt `"Tell me a
joke"` once.

### Advanced Options
The script provides several command-line options to customize the 
evaluation process:

- `-v`, `--verbose`: Enable verbose logging for more detailed output.
- `-m`, `--models`: List of models to evaluate (default: 
`llama3.1:latest`).
- `-p`, `--prompts`: List of prompts to evaluate (default: `"Tell me a joke"`).
- `-r`, `--repeats`: Number of times to repeat each prompt for evaluation 
(default: 1).

#### Example Commands

1. **Evaluate multiple models and prompts with verbosity**:
    ```bash
    python inference_speed.py --repeats 10 --models qwen2.5:32b --prompts "Tell me a joke" "What is your favorite color?"
    ```

2. **Repeat the evaluation multiple times for each prompt**:
    ```bash
    python inference_speed.py --repeats 3 -p "Generate a story"
    ```

## Output
The script logs the inference speed metrics in terms of tokens per second 
(TPS) for both prompt evaluation and response generation. The output is 
formatted as a table, and summary statistics (mean TPS) are also provided.

Example output:
```
Model: llama3.1:latest
INFO:   Prompt eval tps  Response tps
0             68.3          38.4
1            154.5          38.0
2            127.3          39.9
3            153.2          38.2

Model: qwen2.5:32b
INFO:    Prompt eval tps  Response tps
0              48.9          10.0
1              83.0           9.6
2              74.6          10.0
3              83.2           9.6
```

## Contributing
Contributions to this project are welcome! Feel free to open issues or 
submit pull requests with improvements or new features.

### How to Contribute
1. Fork the repository.
2. Create a new branch for your feature or bug fix.
3. Make your changes and commit them.
4. Push your changes to your forked repository.
5. Open a pull request against the original repository.

---
