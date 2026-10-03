from .models import MinimalSource
from pathlib import Path


class Generator():
    def __init__(self, root_path: str):
        self.root_path: Path = Path(root_path)

    def load_context(self, min_sources: list[MinimalSource]) -> list[
            dict[str, list[int] | str]]:
        context_chunk: list[dict] = []
        for src in min_sources:
            try:
                path_file = Path(src.file_path)
                all_path = self.root_path / path_file
                with open(all_path, 'r', encoding='utf-8') as file:
                    first = src.first_character_index
                    last = src.last_character_index
                    data_file = file.read()
                    if not (0 <= first < last <= len(data_file)):
                        raise ValueError("we cant accept invalid index")
                    content = data_file[first: last]
            except FileNotFoundError:
                raise FileNotFoundError(f"file not found {src.file_path}")
            except PermissionError:
                raise PermissionError(f"No permission to {src.file_path}")
            fiche_chunk = {
                "source": src.file_path,
                "character_range": [first, last],
                "content": content
            }
            context_chunk.append(fiche_chunk)
        return context_chunk

    def prompt_construction(self, query: str, context_record: list[
            dict[str, list[int] | str]]):
        prompt = 'Answer only from the provided context.'
        prompt += 'If the context is insufficient, say so. '
        prompt += 'Do not invent facts.\n'
        for i, context in enumerate(context_record, 1):
            head = f"context {i}\n"
            prompt += head
            source = f"source: {context['source']}\n"
            prompt += source
            range_char = f"character range: {context['character_range']}\n"
            prompt += range_char
            content = f"content:\n{context['content']}\n"
            prompt += content
        prompt += f'Question:\n{query}\n'
        prompt += 'Answer:\n'
        return prompt
