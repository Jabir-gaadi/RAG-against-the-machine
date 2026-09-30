from .models import RagDataset, MinimalSource, StudentSearchResults
from .models import MinimalSearchResults
from json import JSONDecodeError
import json
from pathlib import Path
from .models import IndexedChunk
from pydantic import ValidationError
import re
from rank_bm25 import BM25Okapi


class Retriever():
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.loaded_list: list[IndexedChunk] = []
        self.tokenized_corpus: list[list[str]] = []
        self.bm25: BM25Okapi | None = None
        self.build_bm25()

    def load_and_convert(self) -> list[IndexedChunk]:
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

    def tokenize(self, content: str) -> list[str]:
        content = content.lower()
        tokenize_list = re.findall(r'\w+', content)
        return tokenize_list

    def build_bm25(self) -> None:
        self.tokenized_corpus = []
        load_indexed = self.load_and_convert()
        if not load_indexed:
            raise ValueError("There is no chunks to tokenize")
        for chunk in load_indexed:
            self.tokenized_corpus.append(self.tokenize(chunk.content))
        our_bm25 = BM25Okapi(self.tokenized_corpus)
        self.bm25 = our_bm25

    def search(self, query: str, k: int) -> list[MinimalSource]:
        if k <= 0:
            raise ValueError("NUmber of score should > 0 !")
        minimal_src_list: list[MinimalSource] = []
        if k > len(self.loaded_list):
            k = len(self.loaded_list)
        chunk_query = self.tokenize(query)
        if not chunk_query:
            raise ValueError("we get error in chunking the query!")
        if self.bm25 is None:
            self.build_bm25()
        if self.bm25 is None:
            raise RuntimeError("BM25 could not be initialized!")
        score_of_query = self.bm25.get_scores(chunk_query)
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
        searching_quesions: list[MinimalSource] = []
        searching_results: list[MinimalSearchResults] = []
        for data in rag_data.rag_questions:
            searching_quesions = self.search(data.question, k)
            searching_results.append(MinimalSearchResults(
                question_id=data.question_id,
                question=data.question,
                retrieved_sources=searching_quesions
            ))
        return StudentSearchResults(search_results=searching_results, k=k)
