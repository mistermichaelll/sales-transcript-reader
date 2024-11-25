from src.sales_transcript_reader.app.anthropic import (
    DEFAULT_ANTHROPIC_MODEL,
    Anthropic,
    run_transcript_eval,
)

from langchain_anthropic import ChatAnthropic

TRANSCRIPT_READER_PATH = "src/sales_transcript_reader/"

a = Anthropic()

run_transcript_eval(
    a,
    TRANSCRIPT_READER_PATH + "prompts/objections.txt",
    "tests/fake-transcripts/objections.txt",
    ChatAnthropic(model=DEFAULT_ANTHROPIC_MODEL, temperature=0.0),
)
