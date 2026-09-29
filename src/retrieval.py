from json import JSONDecodeError
import json
from pathlib import Path
from .models import IndexedChunk
from pydantic import ValidationError


class Retriever():
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.loaded_list = []

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

    def tekonizer_algo()
