from dataclasses import dataclass
import ast


@dataclass
class ChunkStructure():
    content: str
    first_character_index: int
    last_character_index: int


def safe_splitter(content: str, start: int, end: int, max_chunk_size: int):
    if max_chunk_size <= 0:
        raise ValueError("max chunk size always should greather than 0")
    if start >= end:
        return []
    chunks_list = []
    while True:
        if end - start <= max_chunk_size:
            chunk_text = content[start:end]
            # print(chunk_text)
            chunks_list.append(ChunkStructure(
                content=chunk_text,
                first_character_index=start,
                last_character_index=end
                ))
            break
        limit_index = start + max_chunk_size
        split_at_index = content.rfind('\n', start, limit_index)
        if split_at_index == -1 or split_at_index - start < 2:
            split_at_index = content.rfind(' ', start, limit_index)
        if split_at_index == -1 or split_at_index - start < 2:
            chunk_end = limit_index
            chunk_start = limit_index
        else:
            chunk_end = split_at_index + 1
            chunk_start = split_at_index + 1
        chunk_text = content[start:chunk_end]
        chunks_list.append(ChunkStructure(
                content=chunk_text,
                first_character_index=start,
                last_character_index=chunk_end
                ))
        start = chunk_start
    return chunks_list


def chunk_md_text(content: str, max_chunk_size: int):
    len_content = len(content)
    if len_content == 0:
        return []
    if max_chunk_size <= 0:
        raise ValueError("max chunk size always should greather than 0")
    valid_chunks_text = []
    lines = content.splitlines(keepends=True)

    start = 0
    current_index = 0
    for line in lines:
        if line.startswith('#') and current_index > start:
            if current_index - start > max_chunk_size:
                valid_chunks_text += safe_splitter(
                    content,
                    start,
                    current_index,
                    max_chunk_size)
            else:
                valid_chunks_text.append(ChunkStructure(
                    content[start:current_index],
                    start,
                    current_index
                    ))
            start = current_index
        current_index += len(line)
    if current_index - start <= max_chunk_size:
        valid_chunks_text.append(ChunkStructure(
            content[start:current_index],
            start,
            current_index
        ))
    else:
        valid_chunks_text += safe_splitter(
            content,
            start,
            current_index,
            max_chunk_size)
    return valid_chunks_text


def chunk_python_files(content: str, max_chunk_size: int):
    len_content = len(content)
    if len_content == 0:
        return []
    if max_chunk_size <= 0:
        raise ValueError("max chunk size always should greather than 0")
    valid_chunk_py = []
    lines = content.splitlines(keepends=True)

    start = 0
    current_index = 0
    tree = ast.parse(content)
    for section in tree.body:
        if isinstance(section, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)) and current_index > start:
            start_at_line = section.lineno
            end_at_line = section.end_lineno
            start_index = sum(len(li) for li in lines[:start_at_line - 1])
            end_index = sum(len(line) for line in lines[:end_at_line])
