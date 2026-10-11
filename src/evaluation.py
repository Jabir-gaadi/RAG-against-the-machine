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


def evaluation(
    student_search_res_path: str | Path, dataset_path: str | Path
        ) -> dict[int, float]:

    search_path = Path(student_search_res_path)
    dataset_path = Path(dataset_path)
    student_search_data = read_json_files(search_path)
    dataset_content = read_json_files(dataset_path)
    search_data = StudentSearchResults.model_validate(student_search_data)
    rag_datasets = RagDataset.model_validate(dataset_content)
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
            target_source = truth_data_dic[search_question_id][0]
            for one_search_source in search_source[:key]:
                if one_search_source.file_path == target_source.file_path:
                    search_start = one_search_source.first_character_index
                    search_end = one_search_source.last_character_index
                    retrieve_start = target_source.first_character_index
                    retrieve_end = target_source.last_character_index
                    overlap = (
                        retrieve_start < search_end
                        and search_start < retrieve_end
                    )
                    if overlap:
                        hit += 1
                        break
        recall_dic[key] = hit / total_answered_q
    print(recall_dic)
    return recall_dic
