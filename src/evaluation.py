from .models import MinimalSearchResults
from .models import AnsweredQuestion
from pathlib import Path
import json
from json import JSONDecodeError
from .models import StudentSearchResults, RagDataset


def read_json_files(path: Path) -> dict:
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


def evaluation(student_search_res_path: str | Path, dataset_path: str | Path):
    student_search_res_path = Path(student_search_res_path)
    dataset_path = Path(dataset_path)
    json_res = read_json_files(student_search_res_path)
    student_search_res = StudentSearchResults.model_validate(json_res)
    json_data = read_json_files(dataset_path)
    dataset: RagDataset = RagDataset.model_validate(json_data)
    answered_question: list[AnsweredQuestion] = [
        one
        for one in dataset.rag_questions if isinstance(one, AnsweredQuestion)]
    if not answered_question:
        raise ValueError("There is no answered questions")
    truth_question_id: list[str] = [id.question_id for id in answered_question]
    minimal_src_res: list[MinimalSearchResults] = [
        item for item in student_search_res.search_results]
    search_result_id: list[str] = [id.question_id for id in minimal_src_res]
    ground_truth = dict(zip(truth_question_id, answered_question))
    student_result = dict(zip(search_result_id, minimal_src_res))
    k = student_search_res.k

    hits = 0
    total = 0
    for original_question in answered_question:
        question_id = original_question.question_id
        if question_id in ground_truth and question_id in student_result:
            first_k = student_result[question_id].retrieved_sources[:k]
            for retr_source in first_k:
                if retr_source in ground_truth[question_id].sources:
                    hits += 1
                    break
        total += 1
    eval_result = {
        'matched_questions': hits,
        'total_questions': total,
        'recall_at_k': hits / total
    }
    return eval_result
