from models import StudentSearchResults
from models import RagDataset
from json import JSONDecodeError
import json
from .retrieval import Retriever
from pathlib import Path


def run_search(
    data_path: str,
    k: int,
    index_path: str,
    output_path: Path | str
        ):
    try:
        with open(data_path, 'r', encoding='UTF-8') as file:
            json_data = json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"cant found this file {data_path}")
    except PermissionError:
        raise PermissionError(f"No permission to {data_path}")
    except JSONDecodeError:
        raise ValueError("Invalid JSON index file")
    dataset = RagDataset.model_validate(json_data)
    retriever = Retriever(index_path)
    student_search_res: StudentSearchResults = retriever.search_dataset(
                                                                dataset, k)
    try:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as file:
            search_res = json.dumps(student_search_res)
            file.write(search_res)
    except FileNotFoundError:
        raise FileNotFoundError(f'Can;t find : {output_path}')
    except PermissionError:
        raise PermissionError(f'NO permission: {output_path}')
