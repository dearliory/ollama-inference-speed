# Measure the inference speed of Ollama models by evaluating their perofrmance on specified prompts.
#
# Examples:
#   python inference_speed.py --verbose -r 2
#   python inference_speed.py --repeats 10 --models qwen2.5:32b --prompts "What color is the sky" "Write a report on the financials of Microsoft"

import argparse
import collections
import dataclasses
from datetime import datetime
import ollama
import pandas as pd
import logging

_SEC_TO_NANOSEC = 1000000000

# Setting up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclasses.dataclass
class Request:
    """Class for the request prompt fields.
    
    Arguments:
        model: (required) the model name.
        prompt: the prompt to generate a response for.
        suffix: the text after the model response.
        images: (optional) a list of base64-encoded images for multimodal models
            such as llava.
    """
    model: str
    prompt: str
    suffix: str = ""
    images: list[str] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class Message:
    """"""
    role: str
    content: str
    images: list[str] = dataclasses.field(default_factory=list)
    tool_calls: list[dict] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Response:
    """The response message from Ollama.

    Arguments:
        total_duration: time spent generating the response.
        load_duration: time spent in nanoseconds loading the model.
        prompt_eval_count: number of tokens in the prompt.
        prompt_eval_duration: time spent in nanoseconds evaluating the prompt.
        eval_count: number of tokens in the response.
        eval_duration: time in nanoseconds spent generating the response.
        context: an encoding of the conversation used in this response, this can
            be sent in the next request to keep a conversational memory.
        response: empty if the response was streamed, if not streamed, this will
            contain the full response.
    """
    model: str
    created_at: datetime
    message: Message
    done: bool
    context: list = dataclasses.field(default_factory=list)
    total_duration: int = 0
    load_duration: int = 0
    prompt_eval_count: int = 0
    prompt_eval_duration: int = 0
    eval_count: int = 0
    eval_duration: int = 0


def get_response(model: str, prompt: str, verbose: bool) -> Response:
    """Executes the evaluation process and obtain the measurements."""
    if verbose:
        stream = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        for chunk in stream:
            print(chunk["message"]["content"], end="", flush=True)
            response = chunk
    else:
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )

    if not response:
        logger.error("Ollama response is empty.")
        raise

    return response


def get_measurement(response: Response) -> dict[str, str]:
    """Gets the measurement from the Ollama response."""
    prompt_tps = response.prompt_eval_count / (
        response.prompt_eval_duration / _SEC_TO_NANOSEC)
    
    response_tps = response.eval_count / (
        response.eval_duration / _SEC_TO_NANOSEC)

    total_tps = (response.prompt_eval_count + response.eval_count) / (
            response.prompt_eval_duration + response.eval_duration
        ) * _SEC_TO_NANOSEC

    return {
        "model": response.model,
        "Prompt eval tps": prompt_tps,
        "Response tps": response_tps,
        "Total tps": total_tps,
        "Prompt token count": response.prompt_eval_count,
        "Response token count": response.eval_count,
        "Model load time sec": response.load_duration / _SEC_TO_NANOSEC,
        "Prompt eval time sec": response.prompt_eval_duration / _SEC_TO_NANOSEC,
        "Response time sec": response.eval_duration / _SEC_TO_NANOSEC,
        "Total time sec": response.total_duration / _SEC_TO_NANOSEC,
    }


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Measure the inference speed on Ollama models."
    )

    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Increase the verbosity of outputs.",
        default=False,
    )

    parser.add_argument(
        "-m",
        "--models",
        nargs="*",
        default=["llama3.1:latest"],
        help=("List of model to evaluate."),
    )

    parser.add_argument(
        "-p",
        "--prompts",
        nargs="*",
        default=["Tell me a joke"],
        help=("List of prompts to evaluate."),
    )

    parser.add_argument(
        "-r",
        "--repeats",
        type=int,
        default=1,
        help=("Number of prompts to be repeated during evaluation."),
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    logger.info(
        f"Verbose: {args.verbose}\n"
        f"Models: {args.models}\n"
        f"Prompts: {args.prompts}\n"
        f"Repeats: {args.repeats}\n"
    )

    measures = collections.defaultdict(list)

    for model in args.models:
        for index in range(args.repeats):
            for prompt in args.prompts:
                logger.info(f"[{index}]Prompt: {prompt}")
                response = get_response(model, prompt, verbose=args.verbose)
                measures[model].append(get_measurement(response))

    for model, measure in measures.items():
        df = pd.DataFrame(measure)
        df_rounded = df.round(1)
        logger.info(f"model: {model}")
        logger.info(df_rounded.to_json(orient='index', indent=2))
        logger.info(df_rounded[["Prompt eval tps", "Response tps"]])


if __name__ == "__main__":
    main()