from .models import StudentSearchResults
from .models import RagDataset
from json import JSONDecodeError
import json
from .retrieval import Retriever
from .evaluation import evaluation
from pathlib import Path


def run_search(
    data_path: str | Path,
    k: int,
    save_dir: Path | str,
    index_path: Path | str = "data/processed/index.json"
        ) -> StudentSearchResults:
    try:
        data_path = Path(data_path)
        with open(data_path, 'r', encoding='UTF-8') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"cant found this file {data_path}")
    except PermissionError:
        raise PermissionError(f"No permission to {data_path}")
    except JSONDecodeError:
        raise ValueError("Invalid dataset json file")
    dataset = RagDataset.model_validate(json_data)
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
    return run_search(
        data_path=dataset_path, k=k, save_dir=save_directory,
        index_path=index_path
            )


def evaluate(
    student_search_results_path: str | Path,
    dataset_path: str | Path
        ) -> dict[str, int | float]:
    metrics = evaluation(
        student_search_res_path=student_search_results_path,
        dataset_path=dataset_path)
    return metrics
