import json
from pathlib import Path
from .indexing import IndexedChunk


class reatrieval():
    def __init__(self, path: str | Path) -> list[IndexedChunk]:
        self.path = Path(path)
        self.loaded_list = []

    def load_and_convert(self):
        try:
            with open(self.path, 'r', encoding='utf-8') as file:
                dict_data = json.load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"cant found this file {self.path}")
        except PermissionError:
            raise PermissionError(f"No permission to {self.path}")
        for dic in dict_data:
            dic = IndexedChunk(*dic)
