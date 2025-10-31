import os
import time
from typing import List

import numpy as np
from oe_eval.metrics.metric import Metric

from offline_evals.metrics.llm import LanguageModel

try:
    import openai
    from openai import OpenAI

    # set the environement OPENAI_API_KEY = OAI_KEY
    os.environ["OPENAI_API_KEY"] = os.environ.get("OAI_KEY", "")
except Exception:
    pass

from offline_evals.metrics.news_judge_prompt import EVAL_PROMPT as NEWS_EVAL_PROMPT
from offline_evals.metrics.poem_judge_prompt import EVAL_PROMPT as POEM_EVAL_PROMPT


class ChatCompletionSampler:
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key_name = os.environ.get("OAI_KEY")
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self, image: str, encoding: str = "base64", format: str = "png", fovea: int = 768
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content):
        return {"role": str(role), "content": content}

    def __call__(self, queries, **kwargs) -> List[str]:
        messages = [  # noqa: F841
            {
                "role": "user",
                "content": query,
            }
            for query in queries
        ]

        if self.system_message:
            # TODO: should fix F821
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list  # noqa:F821
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return [choice.message.content for choice in response.choices]
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return []
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception


class LMJudgeMetric(Metric):
    """
    Uses a language model to evaluate generated text based on specified criteria.
    Returns a single normalized score between 0 and 1.
    """

    def __init__(
        self,
        model_name: str,
        eval_prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.metric_names = ["lm_score"]

        if model_name.startswith("gpt"):
            # Initialize the LM evaluator
            self.evaluator = ChatCompletionSampler(
                model=model_name, temperature=temperature, max_tokens=max_tokens
            )
        else:
            self.evaluator = LanguageModel(model_name=model_name, max_input_len=max_tokens)

        # Use default evaluation prompt if none provided
        if eval_prompt == "news":
            self.eval_prompt = NEWS_EVAL_PROMPT
        elif eval_prompt == "story":
            self.eval_prompt = POEM_EVAL_PROMPT  # FIXME
        elif eval_prompt == "poem":
            self.eval_prompt = POEM_EVAL_PROMPT  # FIXME
        else:
            raise NotImplementedError()

    def _parse_evaluation(self, evaluation):
        # Extract the total score (expected to be between 0-5)
        if "Total score:" in evaluation:
            # Fix the split to use "Score:" instead of "News article score:"
            texts = evaluation.split("Total score:")
            score_text = texts[1].strip()
            # Extract first number found in the remaining text
            import re

            score_match = re.search(r"(\d+(?:\.\d+)?)", score_text)
            if score_match:
                total_score = float(score_match.group(1))
            else:
                total_score = None
        return total_score

    def process_one_doc(self, group_lst) -> dict:
        """Process a single document's results."""

        outputs = []
        for out in group_lst:
            reference = out["label"]
            generated = out["doc"]["prefix"] + " " + out["model_resps"]["continuation"]
            metadata = out["doc"].get("metadata", None)

            if metadata:
                messages = [
                    self.eval_prompt.format(
                        metadata=metadata, reference=reference, generated=generated
                    )
                ]
            else:
                messages = [self.eval_prompt.format(reference=reference, generated=generated)]

            evaluation = self.evaluator(messages, stop="\n\n")[0]
            total_score = self._parse_evaluation(evaluation)

            outputs.append(
                {
                    "lm_score": min(
                        max((total_score if total_score is not None else 3) / 5.0, 0.0), 1.0
                    ),
                    "raw_judge_output": evaluation,
                }
            )

        return {
            "lm_score": np.mean([output["lm_score"] for output in outputs]),
            "all_outputs": outputs,
        }
