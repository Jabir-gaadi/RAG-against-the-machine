from chunking import chunk_files
from pathlib import Path
import pathlib
import warnings
from .models import IndexedChunk


# je suis
class Indexer:
    def __init__(self, data_path: str | pathlib.Path):
        self.data_path = pathlib.Path(data_path)
        self.documents = {}

    def extract_corpus(self):
        files = self.data_path.rglob("*")
        valid_files = ['.py', '.md', '.txt', '.rst']
        files_list = [
            file for file in files if file.suffix.lower() in valid_files
            ]
        files_list.sort()
        for file in files_list:
            try:
                with open(file, 'r', encoding='utf-8') as f:
                    source = f.read()
            except UnicodeDecodeError:
                warnings.warn("This is not a python file or markdown")
                continue
            except PermissionError:
                warnings.warn(f"This file {file} dosn't have permission")
                continue
            except FileNotFoundError:
                warnings.warn(f"Can't find this file {file}")
                continue
            if file.suffix.lower() == '.py':
                file_type = "python"
            else:
                file_type = "text"
            tmp_path = str(file.relative_to(self.data_path))
            tmp_path = "data/raw/" + tmp_path
            self.documents[tmp_path] = (source, file_type)
        return self.documents

    def main_machine_chunk(self, output_path: Path, max_chunk_size: int):
        documment = self.extract_corpus()
        all_index_records: list[IndexedChunk] = []
        for path, info in documment.items():
            chunk_list = chunk_files(info[0], info[1], max_chunk_size)
            for chunk in chunk_list:
                all_index_records.append(IndexedChunk(
                    content=chunk.content,
                    file_path=path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                    file_type=info[1]))
