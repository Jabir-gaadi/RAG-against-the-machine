from pydantic import ValidationError
from typing import Any
from .models import StudentSearchResultsAndAnswer
from .models import MinimalAnswer
from .generation import Generator
from .models import UnansweredQuestion, RagDataset
from .models import StudentSearchResults, MinimalSource
import json
from .retrieval import Retriever
from .evaluation import evaluation
from .indexing import Indexer
from pathlib import Path
from tqdm import tqdm


def read_json_file(path: str | Path) -> Any:
    """Read and decode a UTF-8 JSON file.

    Args:
        path: Location of the JSON file.

    Returns:
        Decoded JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
        ValueError: If the JSON content is invalid.
    """
    try:
        path = Path(path)
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"cant found this file {path}") from None
    except PermissionError:
        raise PermissionError(f"No permission to {path}")
    except json.JSONDecodeError:
        raise ValueError("Invalid dataset json file")
    return data


def run_search(
    data_path: str | Path,
    k: int,
    save_dir: Path | str,
        index_path: Path | str
        ) -> StudentSearchResults:
    """Search every question in a dataset and save the results as JSON.

    Args:
        data_path: Input RAG dataset JSON path.
        k: Number of sources to retrieve for each question.
        save_dir: Directory where search results are saved.
        index_path: Path to the processed chunk index.

    Returns:
        Validated search results for every dataset question.
    """
    data_path = Path(data_path)
    json_data = read_json_file(data_path)
    try:
        dataset = RagDataset.model_validate(json_data)
    except ValidationError as e:
        first_error = e.errors()[0]
        message = first_error["msg"]
        raise ValueError(f"Invalid dataset : {message}") from None
    retriever = Retriever(index_path)
    student_search_res: StudentSearchResults = retriever.search_dataset(
                                                                    dataset, k)
    try:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / data_path.name
        with open(output_file, 'w', encoding='utf-8') as file:
            search_res = json.dumps(student_search_res.model_dump())
            file.write(search_res)
    except FileNotFoundError:
        raise FileNotFoundError(f'Cant find : {output_file}')
    except PermissionError:
        raise PermissionError(f'NO permission: {output_file}')
    return student_search_res


def search_dataset(
    dataset_path: Path | str,
    k: int,
    save_directory: Path,
        index_path: Path | str = "data/processed/index.json"
        ) -> StudentSearchResults:
    """Fire command that searches a dataset using the configured BM25 index.

    Args:
        dataset_path: Input RAG dataset JSON path.
        k: Number of sources to retrieve for each question.
        save_directory: Directory where the results JSON is written.
        index_path: Path to the processed chunk index.

    Returns:
        Search results saved to ``save_directory``.
    """
    return run_search(
        data_path=dataset_path, k=k, save_dir=save_directory,
        index_path=index_path
            )


def evaluate(
    student_search_results_path: str | Path,
        dataset_path: str | Path
        ) -> None:
    """Print Recall@k metrics for saved search results.

    Args:
        student_search_results_path: Path to student search-results JSON.
        dataset_path: Path to the answered ground-truth dataset JSON.
    """
    evaluation(
        student_search_res_path=student_search_results_path,
        dataset_path=dataset_path)
    return None


def index(
    max_chunk_size: int = 2000,
    raw_data_path: str | Path = "data/raw",
        output_path: str | Path = "data/processed/index.json"
        ) -> dict[str, int | str | Path]:
    """Build and save the searchable index for the raw corpus.

    Args:
        max_chunk_size: Maximum number of characters in one chunk.
        raw_data_path: Directory containing the raw vLLM corpus.
        output_path: Destination path for the index JSON file.

    Returns:
        The saved index path and number of generated chunks.
    """
    raw_data_path = Path(raw_data_path)
    if not raw_data_path.is_dir():
        raise ValueError(f"Error: Cannot find directory: {raw_data_path}")
    indexer_class = Indexer(raw_data_path)
    res_of_index = indexer_class.main_machine_chunk(
        output_path=output_path,
        max_chunk_size=max_chunk_size)
    prepare_result = {
        "index_path": output_path,
        "chunk_count": len(res_of_index)
    }
    return prepare_result


def search(
    query: str,
    k: int,
        index_path: str = "data/processed/index.json"
        ) -> list[MinimalSource] | None:
    """Retrieve the top source locations for one query.

    Args:
        query: Natural-language question to search for.
        k: Number of sources to retrieve.
        index_path: Path to the processed chunk index.

    Returns:
        Top-k source locations ranked by BM25.
    """
    if not query:
        print("query must contain at least one searchable token")
        return None
    retriever_class = Retriever(path=index_path)
    res_of_search = retriever_class.search(query=query, k=k)
    return res_of_search


def answer(
    query: str,
    k: int,
    index_path: str | Path = "data/processed/index.json",
        root_path: str | Path = "."
        ) -> MinimalAnswer:
    """Retrieve evidence and generate one grounded answer.

    Args:
        query: Natural-language question to answer.
        k: Number of source chunks to retrieve.
        index_path: Path to the processed chunk index.
        root_path: Project root used to locate source files.

    Returns:
        A validated answer with the retrieved source locations.
    """
    unanswered_query = UnansweredQuestion(question=query)
    retriver_class = Retriever(index_path)
    min_sources: list[MinimalSource] = retriver_class.search(query, k)
    generator = Generator(root_path)
    generated_answer = generator.answer_question(query, min_sources)
    return MinimalAnswer(
        question_id=unanswered_query.question_id,
        question=query,
        retrieved_sources=min_sources,
        answer=generated_answer
    )


def answer_dataset(
    student_search_results_path: str | Path,
    save_directory: str | Path,
        root_path: str | Path = "."
        ) -> StudentSearchResultsAndAnswer:
    """Generate and save grounded answers for existing search results.

    Args:
        student_search_results_path: Input search-results JSON path.
        save_directory: Directory where the answer-results JSON is written.
        root_path: Project root used to locate source files.

    Returns:
        Answer results containing the original sources and generated answers.
    """
    answers: list[MinimalAnswer] = []
    search_results_path = Path(student_search_results_path)
    search_results_data = read_json_file(search_results_path)
    try:
        student_res = StudentSearchResults.model_validate(search_results_data)
    except ValidationError as e:
        first_error = e.errors()[0]
        message = first_error['msg']
        raise ValueError(f"invalid data : {message}") from None
    generator = Generator(root_path)
    for res in tqdm(student_res.search_results, desc="Generating answers"):
        generated_answer = generator.answer_question(
            res.question, res.retrieved_sources)
        answers.append(MinimalAnswer(
            question_id=res.question_id,
            question=res.question,
            retrieved_sources=res.retrieved_sources,
            answer=generated_answer
        ))
    student_search_answer = StudentSearchResultsAndAnswer(
        search_results=answers,
        k=student_res.k
    )
    try:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        output_file = save_dir / search_results_path.name
        with open(output_file, 'w', encoding='utf-8') as file:
            search_res = json.dumps(student_search_answer.model_dump())
            file.write(search_res)
    except FileNotFoundError:
        raise FileNotFoundError(f'Cant find : {output_file}')
    except PermissionError:
        raise PermissionError(f'NO permission: {output_file}')

    return student_search_answer
