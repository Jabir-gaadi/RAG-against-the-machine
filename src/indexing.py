from .chunking import chunk_files
from pathlib import Path
import pathlib
import warnings
from .models import IndexedChunk
import json
from tqdm import tqdm


class Indexer:
    """Read supported corpus files and create validated chunk index records."""

    def __init__(self, data_path: str | pathlib.Path):
        """Initialize an indexer for a raw corpus directory.

        Args:
            data_path: Root directory containing source files to index.
        """
        self.data_path = pathlib.Path(data_path)
        self.documents: dict[str, tuple[str, str]] = {}

    def extract_corpus(self) -> dict[str, tuple[str, str]]:
        """Read supported source files and preserve corpus-relative paths.

        Returns:
            A mapping from evaluator-compatible file paths to source content
            and file types.

        Warns:
            UserWarning: If a file cannot be decoded, read, or found.
        """
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

    def main_machine_chunk(
        self,
        output_path: str | Path,
        max_chunk_size: int = 2000
            ) -> list[IndexedChunk]:
        """Chunk the corpus and save a UTF-8 JSON index file.

        Args:
            output_path: Destination path for the generated index JSON file.
            max_chunk_size: Maximum number of characters in each chunk.

        Returns:
            Validated index records written to ``output_path``.

        Raises:
            FileNotFoundError: If the output destination cannot be found.
            PermissionError: If the output file cannot be written.
        """
        output_path = Path(output_path)
        documment = self.extract_corpus()
        all_index_records: list[IndexedChunk] = []
        for path, info in tqdm(documment.items()):
            chunk_list = chunk_files(info[0], path, max_chunk_size)
            for chunk in chunk_list:
                tmp_chunk = IndexedChunk(
                    content=chunk.content,
                    file_path=path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                    file_type=info[1])
                all_index_records.append(tmp_chunk)
        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as file:
                json_content = json.dumps([
                    index.model_dump() for index in all_index_records])
                file.write(json_content)
        except FileNotFoundError:
            raise FileNotFoundError(f'Cant find : {output_path}')
        except PermissionError:
            raise PermissionError(f'NO permission: {output_path}')
        return all_index_records
