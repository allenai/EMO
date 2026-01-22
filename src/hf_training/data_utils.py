"""
Utilities for loading and formatting HuggingFace datasets for finetuning.

Key features:
- Load datasets from HF Hub (GSM8K, MMLU, SQuAD, CoQA)
- Format examples for causal LM finetuning (prompt + answer)
- Apply delimiter-based masking: mask all tokens before delimiter with -100 in labels
- Support different task types: multiple choice, generation, QA
"""

import logging
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset

from offline_evals.run_eval import load_task
from scripts.eval.tasks import get_task_configs

logger = logging.getLogger(__name__)

# Task configurations for HuggingFace datasets
TASK_CONFIGS = {
    "gsm8k": {
        "hf_path": "gsm8k",
        "hf_name": "main",
        "prompt_template": "Question: {question}\nAnswer:",
        "answer_field": "answer",
        "delimiter": "Answer:",
        "task_type": "generation",
    },
    "mmlu": {
        "hf_path": "cais/mmlu",
        "hf_name": "all",
        "prompt_template": "{question}\nA. {A}\nB. {B}\nC. {C}\nD. {D}\nAnswer:",
        "answer_field": "answer",  # This is the index (0-3), need to map to letter
        "delimiter": "Answer:",
        "task_type": "multiple_choice",
    },
    "squad": {
        "hf_path": "rajpurkar/squad",
        "hf_name": None,
        "prompt_template": "Context: {context}\nQuestion: {question}\nA:",
        "answer_field": "answers",  # answers is a dict with 'text' list
        "delimiter": "A:",
        "task_type": "extractive_qa",
    },
    "coqa": {
        "hf_path": "stanfordnlp/coqa",
        "hf_name": None,
        "prompt_template": "Context: {story}\nQuestion: {question}\nAnswer:",
        "answer_field": "answer",
        "delimiter": "Answer:",
        "task_type": "conversational_qa",
    },
}

# Map MMLU answer indices to letters
MMLU_ANSWER_MAP = {0: "A", 1: "B", 2: "C", 3: "D"}


def get_task_config(task_name: str) -> dict:
    """Get task configuration, handling task variants like 'gsm8k_train'."""
    # Extract base task name
    base_task = task_name.split("_")[0].lower()
    if base_task not in TASK_CONFIGS:
        raise ValueError(f"Unknown task: {task_name}. Supported tasks: {list(TASK_CONFIGS.keys())}")
    return TASK_CONFIGS[base_task]


def load_hf_dataset(task_name: str, split: str) -> Dataset:
    """
    Load a dataset from HuggingFace Hub.

    Args:
        task_name: Name of the task (gsm8k, mmlu, squad, coqa)
        split: Dataset split (train, validation, test)

    Returns:
        HuggingFace Dataset
    """
    config = get_task_config(task_name)

    logger.info(f"Loading dataset {config['hf_path']} ({config.get('hf_name', 'default')}) split={split}")

    try:
        if config["hf_name"]:
            dataset = load_dataset(config["hf_path"], config["hf_name"], split=split)
        else:
            dataset = load_dataset(config["hf_path"], split=split)
    except ValueError as e:
        # Handle split name variations (e.g., gsm8k uses 'test' for validation)
        if "validation" in str(e) and split == "validation":
            logger.warning(f"Validation split not found, trying 'test' split for {task_name}")
            if config["hf_name"]:
                dataset = load_dataset(config["hf_path"], config["hf_name"], split="test")
            else:
                dataset = load_dataset(config["hf_path"], split="test")
        else:
            raise

    logger.info(f"Loaded {len(dataset)} examples from {task_name} {split}")
    return dataset


def format_gsm8k_example(example: dict) -> Tuple[str, str]:
    """Format a GSM8K example into prompt and answer."""
    prompt = f"Question: {example['question']}\nAnswer:"
    # GSM8K answer format: "explanation text #### numeric_answer"
    answer = " " + example["answer"]
    return prompt, answer


def format_mmlu_example(example: dict) -> Tuple[str, str]:
    """Format an MMLU example into prompt and answer."""
    choices = example["choices"]
    prompt = f"{example['question']}\nA. {choices[0]}\nB. {choices[1]}\nC. {choices[2]}\nD. {choices[3]}\nAnswer:"
    # answer is the index (0-3), map to letter
    answer_letter = MMLU_ANSWER_MAP[example["answer"]]
    answer = " " + answer_letter
    return prompt, answer


def format_squad_example(example: dict) -> Tuple[str, str]:
    """Format a SQuAD example into prompt and answer."""
    prompt = f"Context: {example['context']}\nQuestion: {example['question']}\nA:"
    # answers is a dict with 'text' list, take the first answer
    answer_text = example["answers"]["text"][0] if example["answers"]["text"] else ""
    answer = " " + answer_text
    return prompt, answer


def format_coqa_example(example: dict, turn_idx: int = -1) -> Tuple[str, str]:
    """
    Format a CoQA example into prompt and answer.

    CoQA has multiple turns of Q&A per story. By default, use the last turn.
    """
    story = example["story"]
    questions = example["questions"]
    answers = example["answers"]

    # Use specified turn or last turn
    idx = turn_idx if turn_idx >= 0 else len(questions) - 1
    idx = min(idx, len(questions) - 1)

    # Build conversation history
    history = ""
    for i in range(idx):
        history += f"Q: {questions[i]}\nA: {answers['input_text'][i]}\n"

    prompt = f"Context: {story}\n{history}Q: {questions[idx]}\nAnswer:"
    answer = " " + answers["input_text"][idx]
    return prompt, answer


def format_example(example: dict, task_name: str) -> Tuple[str, str]:
    """
    Format a single example as prompt + answer text.

    Args:
        example: Raw example from HuggingFace dataset
        task_name: Name of the task

    Returns:
        Tuple of (prompt, answer)
    """
    base_task = task_name.split("_")[0].lower()

    if base_task == "gsm8k":
        return format_gsm8k_example(example)
    elif base_task == "mmlu":
        return format_mmlu_example(example)
    elif base_task == "squad":
        return format_squad_example(example)
    elif base_task == "coqa":
        return format_coqa_example(example)
    else:
        raise ValueError(f"Unknown task: {task_name}")


def find_delimiter_position(input_ids: List[int], delimiter_ids: List[int]) -> int:
    """
    Find the position of the delimiter tokens in input_ids.

    Returns the index of the first token after the delimiter, or -1 if not found.
    """
    for i in range(len(input_ids) - len(delimiter_ids) + 1):
        if input_ids[i : i + len(delimiter_ids)] == delimiter_ids:
            return i + len(delimiter_ids)
    return -1


def create_masked_labels(
    input_ids: List[int],
    delimiter_ids: List[int],
    ignore_index: int = -100,
) -> List[int]:
    """
    Create labels with tokens before delimiter masked.

    For causal LM training, we want to compute loss only on the answer portion.
    All tokens before and including the delimiter are masked with ignore_index.

    Args:
        input_ids: List of token IDs
        delimiter_ids: Token IDs of the delimiter (e.g., "Answer:")
        ignore_index: Value to use for masked positions (default -100)

    Returns:
        List of labels with masked positions
    """
    labels = list(input_ids)  # Copy input_ids

    # Find delimiter position
    delimiter_pos = find_delimiter_position(input_ids, delimiter_ids)

    if delimiter_pos == -1:
        # If delimiter not found, log warning and return all masked
        logger.warning("Delimiter not found in input, masking all tokens")
        return [ignore_index] * len(input_ids)

    # Mask everything before and including the delimiter
    for i in range(delimiter_pos):
        labels[i] = ignore_index

    return labels


def tokenize_and_mask_example(
    prompt: str,
    answer: str,
    tokenizer,
    max_length: int = 4096,
    delimiter: str = "Answer:",
) -> Dict:
    """
    Tokenize prompt+answer and create masked labels.

    Args:
        prompt: The prompt text (question/context)
        answer: The answer text
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length
        delimiter: Delimiter string that separates prompt from answer

    Returns:
        Dict with input_ids, attention_mask, and labels
    """
    # Combine prompt and answer
    full_text = prompt + answer

    # Tokenize
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding=False,
        return_tensors=None,
    )

    input_ids = tokenized["input_ids"]
    attention_mask = tokenized["attention_mask"]

    # Get delimiter token IDs
    delimiter_ids = tokenizer(delimiter, add_special_tokens=False)["input_ids"]

    # Create masked labels
    labels = create_masked_labels(input_ids, delimiter_ids)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def prepare_finetuning_dataset(
    task_name: str,
    split: str,
    tokenizer,
    max_length: int = 4096,
) -> Dataset:
    """
    Load and prepare a HuggingFace dataset for finetuning.

    Args:
        task_name: Name of the task (gsm8k, mmlu, squad, coqa)
        split: Dataset split (train, validation, test)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset with tokenized examples
    """
    config = get_task_config(task_name)

    # Load raw dataset
    raw_dataset = load_hf_dataset(task_name, split)

    # Get delimiter
    delimiter = config["delimiter"]

    def process_example(example):
        prompt, answer = format_example(example, task_name)
        return tokenize_and_mask_example(prompt, answer, tokenizer, max_length, delimiter)

    # Process all examples
    logger.info(f"Tokenizing {len(raw_dataset)} examples...")
    processed_dataset = raw_dataset.map(
        process_example,
        remove_columns=raw_dataset.column_names,
        desc="Tokenizing",
    )

    logger.info(f"Prepared {len(processed_dataset)} examples for finetuning")
    return processed_dataset


def load_finetuning_dataset(task_name: str, split: str, tokenizer, max_length: int = 4096) -> Dataset:
    """
    Load and format HF dataset for finetuning.

    This is the main entry point for loading datasets.

    Args:
        task_name: Name of the task (gsm8k, mmlu, squad, coqa)
        split: Dataset split (train, validation, test)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        HuggingFace Dataset ready for training
    """
    return prepare_finetuning_dataset(task_name, split, tokenizer, max_length)


def get_formatted_prompts(task_name: str, split: str) -> List[str]:
    """
    Get formatted prompts (prompt + answer) for a dataset.

    Useful for computing router activations where we just need the text.

    Args:
        task_name: Name of the task
        split: Dataset split

    Returns:
        List of formatted prompt+answer strings
    """
    breakpoint()
    TASK_CONFIGS = get_task_configs()
    task_config = TASK_CONFIGS[task_name]
    task = load_task(task_config, "tmp")
    task.download()
    task.build_all_requests()

    # TODO: need to filter out the incorrect examples somehow

    dataset = []

    if task._instances[0].request_type == "loglikelihood":

        for instance in task._instances:
            # we only choose the correct instances
            if instance.idx == instance.label and not instance.request.context.startswith("Answer:"):
                dataset.append(instance.request.context + instance.request.continuation)

    elif task._instances[0].request_type == "generate_until":
        raise NotImplementedError("generate_until not implemented in get_formatted_prompts")
        # for req in requests_data:
        #     # for some tasks (e.g coqa), by default there is no space between context and choice, so we add it here
        #     if req["request"]["context"][-1] != " " and req["doc"]["choices"][0][0] != " ":
        #         data += [req["request"]["context"] + " " + req["doc"]["choices"][0]]
        #     else:
        #         data += [req["request"]["context"] + req["doc"]["choices"][0]]

    breakpoint()

    # raw_dataset = load_hf_dataset(task_name, split)
    #
    # prompts = []
    # for example in raw_dataset:
    #     prompt, answer = format_example(example, task_name)
    #     prompts.append(prompt + answer)

    return dataset
