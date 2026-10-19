from typing import Any
from pydantic import ValidationError
from .models import AnsweredQuestion
from pathlib import Path
import json
from json import JSONDecodeError
from .models import StudentSearchResults, RagDataset


def calucule_IoU(start1: int, end1: int, start2: int, end2: int) -> float:
    intersection = min(end1, end2) - max(start1, start2)
    if intersection <= 0:
        return 0.0
    first = end1 - start1
    second = end2 - start2
    union = first + second - intersection
    IoU = intersection / union
    return IoU


def read_json_files(path: Path) -> dict:
    """Read and decode a UTF-8 JSON file.

    Args:
        path: Location of the JSON file to read.

    Returns:
        Decoded JSON content.

    Raises:
        FileNotFoundError: If the file does not exist.
        PermissionError: If the file cannot be read.
        ValueError: If the JSON content is invalid.
    """
    try:
        with open(path, 'r', encoding='utf-8') as file:
            json_res = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"cant found this file {path}")
    except PermissionError:
        raise PermissionError(
            f"No permission to {path}")
    except JSONDecodeError:
        raise ValueError("Invalid json file")
    return json_res


def evaluation(
    student_search_res_path: str | Path, dataset_path: str | Path
        ) -> Any:
    """Calculate Recall@k from search results and answered ground truth.

    A question counts as a hit when a retrieved source has the same file path
    as the expected source and their character ranges overlap.

    Args:
        student_search_res_path: Path to student search-results JSON.
        dataset_path: Path to the answered ground-truth dataset JSON.

    Returns:
        Recall values keyed by the evaluated cutoffs: 1, 3, 5, and 10.
    """

    search_path = Path(student_search_res_path)
    dataset_path = Path(dataset_path)
    student_search_data = read_json_files(search_path)
    dataset_content = read_json_files(dataset_path)
    try:
        search_data = StudentSearchResults.model_validate(student_search_data)
        rag_datasets = RagDataset.model_validate(dataset_content)
    except ValidationError as e:
        first_error = e.errors()[0]
        message = first_error['msg']
        raise ValueError(f"Error : {message}") from None
    truth_data_dic = {
        answer.question_id: answer.sources
        for answer in rag_datasets.rag_questions
        if isinstance(answer, AnsweredQuestion)
        }
    search_data_dic = {
        search.question_id: search.retrieved_sources
        for search in search_data.search_results
        }
    k = search_data.k
    total_answered_q = len(truth_data_dic)
    hit = 0
    recall_dic = {}
    key_val = [1, 3, 5, 10]
    for key in key_val:
        if k < key:
            break
        hit = 0
        for search_question_id, search_source in search_data_dic.items():
            if search_question_id not in truth_data_dic:
                continue
            if truth_data_dic[search_question_id]:
                target_source = truth_data_dic[search_question_id][0]
            else:
                raise ValueError("each rag question must contain\
at least 1 source")
            for one_search_source in search_source[:key]:
                if one_search_source.file_path == target_source.file_path:
                    search_start = one_search_source.first_character_index
                    search_end = one_search_source.last_character_index
                    retrieve_start = target_source.first_character_index
                    retrieve_end = target_source.last_character_index
                    overlap = (
                        calucule_IoU(
                            search_start, search_end,
                            retrieve_start, retrieve_end) > 0.05
                    )
                    if overlap:
                        hit += 1
                        break
        recall_dic[key] = hit / total_answered_q
    print("\nRetrieval evaluation")
    print(f"Questions evaluated: {total_answered_q}")
    for key in key_val:
        if k < key:
            break
        print(f"Recall@{key}:  {recall_dic[key]:.1%}")
