from .models import RagDataset, MinimalSource, StudentSearchResults
from .models import MinimalSearchResults
from json import JSONDecodeError
import json
from pathlib import Path
from .models import IndexedChunk
from pydantic import ValidationError
from re import findall
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class Retriever():
    """Load indexed chunks and retrieve relevant source ranges with BM25.

    Attributes:
        path: Path to the JSON index file.
        loaded_list: Validated chunks loaded from the index.
        tokenized_corpus: Tokenized contents used to build BM25.
        bm25: BM25 model built from the indexed chunks.
    """

    def __init__(self, path: str | Path):
        """Initialize the retriever and build its BM25 index.

        Args:
            path: Path to the JSON file containing indexed chunks.
        """

        self.path = Path(path)
        self.loaded_list: list[IndexedChunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.build_bm25()

    def load_and_convert(self) -> list[IndexedChunk]:
        """Load the JSON index and validate each item as an indexed chunk.

        Returns:
            Validated chunks read from the index file.

        Raises:
            FileNotFoundError: If the index file does not exist.
            PermissionError: If the index file cannot be read.
            ValueError: If the JSON is invalid or an item has an invalid
                schema.
        """

        self.loaded_list = []
        try:
            with open(self.path, 'r', encoding='utf-8') as file:
                dict_data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"cant found this file {self.path}")
        except PermissionError:
            raise PermissionError(f"No permission to {self.path}")
        except JSONDecodeError:
            raise ValueError("Invalid JSON index file")
        for dic in dict_data:
            try:
                self.loaded_list.append(IndexedChunk.model_validate(dic))
            except ValidationError as e:
                raise ValueError(f"invalid chunk !: \n {e}")
        return self.loaded_list

    def tokenize_text(self, content: str) -> list[str]:
        """Convert natural-language text into lowercase word tokens.

        Args:
            content: Text to tokenize.

        Returns:
            Lowercase tokens extracted from the text.
        """

        content = content.lower()
        tokenize_list = findall(r'\w+', content)
        return tokenize_list

    def tokenize_python(self, content: str) -> list[str]:
        """Tokenize Python-related text, preserving and splitting identifiers.

        Args:
            content: Python source text or a Python-related query.

        Returns:
            Lowercase identifier tokens, including underscore-separated parts.
        """

        content = content.lower()
        final_tokens = []
        tokenize_list = findall(r"[A-Za-z0-9_]+", content)
        for chunk in tokenize_list:
            final_tokens.append(chunk)
            if "_" in chunk:
                parts = chunk.split("_")
                final_tokens.extend(part for part in parts if part)
        return final_tokens

    def build_bm25(self) -> None:
        """Load indexed chunks, tokenize them, and construct the BM25 model.

        Raises:
            ValueError: If the index does not contain any chunks.
        """

        self.tokenized_corpus = []
        load_indexed = self.load_and_convert()
        if not load_indexed:
            raise ValueError("There is no chunks to tokenize")
        for chunk in load_indexed:
            if chunk.file_type == "python":
                file_name = Path(chunk.file_path).stem
                to_tokenize = chunk.content + '\n' + file_name
                self.tokenized_corpus.append(self.tokenize_python(to_tokenize))
            else:
                to_tokenize = chunk.content
                self.tokenized_corpus.append(self.tokenize_text(to_tokenize))
        our_bm25 = BM25Okapi(self.tokenized_corpus)
        self.bm25 = our_bm25

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the ``k`` source ranges most relevant to a query.

        Args:
            query: Natural-language question to search for.
            k: Maximum number of source ranges to return.

        Returns:
            Ranked source ranges for the best matching indexed chunks.

        Raises:
            ValueError: If ``k`` is not positive or the query has no tokens.
            RuntimeError: If BM25 cannot be initialized.
        """

        if k <= 0:
            raise ValueError("k must be greater than 0!")
        minimal_src_list: list[MinimalSource] = []
        if k > len(self.loaded_list):
            k = len(self.loaded_list)
        query_tokens = self.tokenize_text(query)
        if not query_tokens:
            raise ValueError("query must contain at least 1 searchable token")
        if self.bm25 is None:
            self.build_bm25()
        if self.bm25 is None:
            raise RuntimeError("BM25 could not be initialized!")
        score_of_query = self.bm25.get_scores(query_tokens)
        enumerate_score = enumerate(score_of_query)
        sorted_score = sorted(
            enumerate_score, key=lambda x: x[1], reverse=True)
        top_k_chunk = sorted_score[:k]
        for best_score in top_k_chunk:
            index = best_score[0]
            chunk = self.loaded_list[index]
            minimal_src_list.append(MinimalSource(
                file_path=chunk.file_path,
                first_character_index=chunk.first_character_index,
                last_character_index=chunk.last_character_index
            ))
        return minimal_src_list

    def search_dataset(
        self,
        rag_data: RagDataset,
        k: int
            ) -> StudentSearchResults:
        """Retrieve sources for every question in a RAG dataset.

        Args:
            rag_data: Dataset containing questions to search.
            k: Maximum number of sources to retrieve per question.

        Returns:
            Search results with the original question metadata and sources.
        """

        searching_quesions: list[MinimalSource] | None = []
        searching_results: list[MinimalSearchResults] = []
        for data in tqdm(rag_data.rag_questions):
            searching_quesions = self.search(data.question, k)
            if not searching_quesions:
                continue
            searching_results.append(MinimalSearchResults(
                question_id=data.question_id,
                question=data.question,
                retrieved_sources=searching_quesions
            ))
        return StudentSearchResults(search_results=searching_results, k=k)
