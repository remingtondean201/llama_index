import logging
from typing import Any, Dict, List, Optional, Sequence

from llama_index.callbacks.schema import CBEventType
from llama_index.data_structs.node import NodeWithScore
from llama_index.evaluation.base import BaseEvaluator
from llama_index.indices.list.base import GPTListIndex
from llama_index.indices.postprocessor.types import BaseNodePostprocessor
from llama_index.indices.query.base import BaseQueryEngine
from llama_index.indices.query.response_synthesis import ResponseSynthesizer
from llama_index.indices.query.schema import QueryBundle
from llama_index.indices.response.type import ResponseMode
from llama_index.indices.service_context import ServiceContext
from llama_index.optimization.optimizer import (
    BaseTokenUsageOptimizer,
    SentenceEmbeddingOptimizer,
)
from llama_index.prompts.prompts import (
    QuestionAnswerPrompt,
    RefinePrompt,
    SimpleInputPrompt,
)
from llama_index.readers.schema.base import Document
from llama_index.response.schema import RESPONSE_TYPE, Response, StreamingResponse

logger = logging.getLogger(__name__)


class RetryQueryEngine(BaseQueryEngine):
    """Retriever query engine with retry.

    Args:
        base_query_engine: BaseQueryEngine.

    """

    def __init__(
        self,
        query_engine: BaseQueryEngine,
        evaluator: BaseEvaluator,
        max_retries: int = 3,
    ) -> None:
        self._query_engine = query_engine
        self._evaluator = evaluator
        self.max_retries = max_retries

    def _query(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Answer a query."""
        response = self._query_engine._query(query_bundle)
        if self.max_retries <= 0:
            return response
        typed_response = (
            response if isinstance(response, Response) else response.get_response()
        )
        query_str = query_bundle.query_str
        eval = self._evaluator.evaluate_response(query_str, typed_response).passing
        if eval:
            logger.debug("Evaluation returned True.")
            return response
        else:
            logger.debug("Evaluation returned False.")
            new_query_engine = RetryQueryEngine(
                self._query_engine, self._evaluator, self.max_retries - 1
            )
            new_query = (
                query_str
                + "\n----------------\n"
                + self.construct_feedback(typed_response.response)
            )
            return new_query_engine.query(new_query)

    async def _aquery(self, query_bundle: QueryBundle) -> RESPONSE_TYPE:
        """Not supported."""
        return self._query(query_bundle)

    def construct_feedback(self, response: Optional[str]) -> str:
        """Construct feedback from response."""
        if response is None:
            return ""
        else:
            return "Here is a previous bad answer.\n" + response
