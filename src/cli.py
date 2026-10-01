from .models import StudentSearchResults
from .models import RagDataset
from json import JSONDecodeError
import json
from .retrieval import Retriever
from pathlib import Path


def run_search(
    data_path: str | Path,
    k: int,
    index_path: Path | str,
    save_dir: Path | str
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
