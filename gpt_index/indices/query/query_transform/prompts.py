"""Query transform prompts."""


from enum import Enum
from typing import List

from gpt_index.prompts.base import Prompt
from gpt_index.prompts.prompt_type import PromptType


class DecomposeQueryTransformPrompt(Prompt):
    """Decompose prompt for query transformation.

    Prompt to "decompose" a query into another query
    given the existing context.

    Required template variables: `context_str`, `query_str`

    Args:
        template (str): Template for the prompt.
        **prompt_kwargs: Keyword arguments for the prompt.

    """

    # TODO: specify a better prompt type
    prompt_type: PromptType = PromptType.CUSTOM
    input_variables: List[str] = ["context_str", "query_str"]


DEFAULT_DECOMPOSE_QUERY_TRANSFORM_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have an opportunity to answer some, or all of the question from a "
    "knowledge source. "
    "Context information for the knowledge source is provided below. \n"
    "Given the context, return a new question that can be answered from "
    "the context. The question can be the same as the original question, "
    "or a new question that represents a subcomponent of the overall question.\n"
    "As an example: "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides information about the winners of the 2020 "
    "Australian Open\n"
    "New question: Who was the winner of the 2020 Australian Open? "
    "\n\n"
    "Question: What is the current population of the city in which Paul Graham found "
    "his first company, Viaweb?\n"
    "Knowledge source context: Provides information about Paul Graham's "
    "professional career, including the startups he's founded. "
    "New question: In which city did Paul Graham found his first company, Viaweb? "
    "\n\n"
    "Question: {query_str}\n"
    "Knowledge source context: {context_str}\n"
    "New question: "
)

DEFAULT_DECOMPOSE_QUERY_TRANSFORM_PROMPT = DecomposeQueryTransformPrompt(
    DEFAULT_DECOMPOSE_QUERY_TRANSFORM_TMPL
)


class CoTDecomposeQueryTransformPrompt(Prompt):
    """CoT Decompose prompt for query transformation.

    Prompt to "decompose" a query into another query
    given the existing context + previous reasoning.

    Required template variables: `context_str`, `query_str`, `prev_reasoning`

    Args:
        template (str): Template for the prompt.
        **prompt_kwargs: Keyword arguments for the prompt.

    """

    # TODO: specify a better prompt type
    prompt_type: PromptType = PromptType.CUSTOM
    input_variables: List[str] = ["context_str", "query_str", "prev_reasoning"]


DEFAULT_COT_DECOMPOSE_QUERY_TRANSFORM_TMPL = (
    "The original question is as follows: {query_str}\n"
    "We have an opportunity to answer some, or all of the question from a "
    "knowledge source. "
    "Context information for the knowledge source is provided below, as "
    "well as previous reasoning steps.\n"
    "Given the context and previous reasoning, return a new question that can "
    "be answered from "
    "the context. The question can be the same as the original question, "
    "or a new question that represents a subcomponent of the overall question.\n"
    "As an example: "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides information about the winners of the 2020 "
    "Australian Open\n"
    "Previous reasoning: None."
    "New question: Who was the winner of the 2020 Australian Open? "
    "\n\n"
    "Question: How many Grand Slam titles does the winner of the 2020 Australian "
    "Open have?\n"
    "Knowledge source context: Provides information about the winners of the 2020 "
    "Australian Open\n"
    "Previous reasoning:\n"
    "- The winner of the 2020 Australian Open was Novak Djokovic.\n"
    "New question: How many Grand Slam titles does Novak Djokovic have? "
    "\n\n"
    "Question: {query_str}\n"
    "Knowledge source context: {context_str}\n"
    "Previous reasoning: \n{prev_reasoning}\n"
    "New question: "
)

DEFAULT_COT_DECOMPOSE_QUERY_TRANSFORM_PROMPT = CoTDecomposeQueryTransformPrompt(
    DEFAULT_COT_DECOMPOSE_QUERY_TRANSFORM_TMPL
)
