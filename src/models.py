from pydantic import BaseModel, Field
import uuid
from typing import List


class MinimalSource(BaseModel):
    """Identify one source range retrieved from the indexed corpus.

    Attributes:
        file_path: Path of the source file.
        first_character_index: Inclusive index where the source range starts.
        last_character_index: Exclusive index where the source range ends.
    """

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Represent a question that still needs a RAG answer.

    Attributes:
        question_id: Unique identifier of the question.
        question: Natural-language question to answer.
    """

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represent a question with its expected sources and answer.

    Attributes:
        sources: Reference source ranges supporting the answer.
        answer: Expected textual answer.
    """

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Store a collection of answered or unanswered RAG questions.

    Attributes:
        rag_questions: Questions contained in the dataset.
    """

    rag_questions: List[AnsweredQuestion | UnansweredQuestion]


class MinimalSearchResults(BaseModel):
    """Store retrieval results for one question.

    Attributes:
        question_id: Identifier of the processed question.
        question: Text of the processed question.
        retrieved_sources: Source ranges returned by the retriever.
    """

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Store a generated answer together with its retrieved sources.

    Attributes:
        answer: Generated response to the question.
    """

    answer: str


class StudentSearchResults(BaseModel):
    """Store all retrieval outputs produced for a dataset.

    Attributes:
        search_results: Retrieval result for each processed question.
        k: Number of sources requested for each question.
    """

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Store generated answers and their retrieval metadata for a dataset.

    Attributes:
        search_results: Generated answer for each processed question.
        k: Number of sources used for each answer.
    """

    search_results: List[MinimalAnswer]
    k: int


class IndexedChunk(BaseModel):
    """Represent one chunk saved in the retrieval index.

    Attributes:
        content: Text contained in the indexed chunk.
        file_path: Path of the file from which the chunk came.
        first_character_index: Inclusive starting character index in the file.
        last_character_index: Exclusive ending character index in the file.
        file_type: Extension of the source file.
    """

    content: str
    file_path: str
    first_character_index: int
    last_character_index: int
    file_type: str
