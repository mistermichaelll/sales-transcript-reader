import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

load_dotenv()

DEFAULT_ANTHROPIC_MODEL = os.environ["ANTHROPIC_DEFAULT_MODEL"]


@dataclass
class Prompt:
    prompt_id: int
    prompt_text: str


class Anthropic:
    def __init__(self):
        self.pricing = {
            "claude-3-5-haiku-20241022": {
                "input_price_per_million": 1,
                "output_price_per_million": 5,
            }
        }

    def setup_default_transcript_prompt(self, prompt_path: str = "") -> "Anthropic":
        """
        Create a ChatPromptTemplate for the default behavior, which is to apply
        some prompt to a sales transcript.
        """
        with open(prompt_path) as prompt_file:
            prompt = prompt_file.read()

        self.prompt = ChatPromptTemplate.from_messages(
            [("system", prompt), ("user", "{transcript}")]
        )

        return self

    def setup_chain(
        self,
        model: Optional[ChatAnthropic] = None,
    ):
        if model is None:
            model = ChatAnthropic(
                model=DEFAULT_ANTHROPIC_MODEL, temperature=0.0
            )  # type: ignore
        self.chain = self.prompt | model

    def invoke_chain(self, transcript_path: str) -> BaseMessage:
        with open(transcript_path) as transcript_file:
            transcript = transcript_file.read()

        return self.chain.invoke({"transcript": transcript})

    def get_total_cost(self, model: str, response: BaseMessage) -> float:
        usage = response.response_metadata["usage"]

        input_cost = (usage.get("input_tokens") / 1_000_000) * self.pricing.get(
            model
        ).get("input_price_per_million")
        output_cost = (usage.get("output_tokens") / 1_000_000) * self.pricing.get(
            model
        ).get("output_price_per_million")

        return input_cost + output_cost


def run_transcript_eval(
    a: Anthropic, prompt_path, transcript_path, model
) -> BaseMessage:
    a.setup_default_transcript_prompt(prompt_path)
    a.setup_chain(model)
    evaluated_transcript = a.invoke_chain(transcript_path)
    return evaluated_transcript
