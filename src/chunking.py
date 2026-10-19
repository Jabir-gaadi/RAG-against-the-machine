from dataclasses import dataclass
import ast
from pathlib import Path


@dataclass
class ChunkStructure():
    """Store a chunk and the character range it occupies in its source file.

    Attributes:
        content: Text contained in the chunk.
        first_character_index: Inclusive start position in the source text.
        last_character_index: Exclusive end position in the source text.
    """

    content: str
    first_character_index: int
    last_character_index: int


def safe_splitter(
    content: str,
    start: int,
        end: int,
        max_chunk_size: int
        ) -> list[ChunkStructure]:
    """Split one text range into chunks that do not exceed the size limit.

    Args:
        content: Full source text containing the requested range.
        start: Inclusive character index where the range starts.
        end: Exclusive character index where the range ends.
        max_chunk_size: Maximum number of characters allowed per chunk.

    Returns:
        Chunks preserving their positions in the original source text.

    Raises:
        ValueError: If ``max_chunk_size`` is not positive.
    """
    if max_chunk_size <= 0:
        raise ValueError("max chunk size always should greather than 0")
    if start >= end:
        return []
    chunks_list = []
    while True:
        if end - start <= max_chunk_size:
            chunk_text = content[start:end]
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


def chunk_md_text(content: str, max_chunk_size: int) -> list[ChunkStructure]:
    """Split Markdown or plain text at headings and safe text boundaries.

    Args:
        content: Markdown or plain-text source content.
        max_chunk_size: Maximum number of characters allowed per chunk.

    Returns:
        Chunks that begin at Markdown headings when possible.

    Raises:
        ValueError: If ``max_chunk_size`` is not positive.
    """
    len_content = len(content)
    if len_content == 0:
        return []
    if max_chunk_size <= 0:
        raise ValueError("max chunk size always should greather than 0")
    valid_chunks_text: list[ChunkStructure] = []
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


def chunk_python_files(
        content: str,
        max_chunk_size: int
        ) -> list[ChunkStructure]:
    """Split Python content around top-level class and function definitions.

    Args:
        content: Python source code to chunk.
        max_chunk_size: Maximum number of characters allowed per chunk.

    Returns:
        Chunks preserving top-level Python semantic boundaries when possible.

    Raises:
        ValueError: If ``max_chunk_size`` is not positive.
    """
    len_content = len(content)
    if len_content == 0:
        return []
    if max_chunk_size <= 0:
        raise ValueError("max chunk size always should greather than 0")
    lines = content.splitlines(keepends=True)
    line_offs = [0]
    for line in lines:
        line_offs.append(line_offs[-1] + len(line))
    buff_start = 0
    valid_chunk_py = []

    def add_range(start: int, end: int) -> None:
        """Append a range directly or split it when it exceeds the limit."""
        if start >= end:
            return
        if end - start <= max_chunk_size:
            if content[start:end].isspace():
                return
            valid_chunk_py.append(ChunkStructure(
                content[start:end],
                start,
                end
            ))
        else:
            valid_chunk_py.extend(safe_splitter(
                content,
                start,
                end,
                max_chunk_size
                    ))
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return safe_splitter(content, 0, len_content, max_chunk_size)
    for section in tree.body:
        if not isinstance(section, (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef
                )):
            continue
        start_line = section.lineno
        start_index = line_offs[start_line - 1]
        end_line = section.end_lineno
        if end_line is None:
            continue
        end_index = line_offs[end_line]
        add_range(buff_start, start_index)
        add_range(start_index, end_index)
        buff_start = end_index
    add_range(buff_start, len_content)
    return valid_chunk_py


def chunk_files(
    content: str,
    file_path: str | Path,
        max_chunk_size: int
        ) -> list[ChunkStructure]:
    """Choose the chunking strategy appropriate for the supplied file path.

    Args:
        content: Source content to chunk.
        file_path: Path used to determine the file type.
        max_chunk_size: Maximum number of characters allowed per chunk.

    Returns:
        Chunks produced by the Python or text chunking strategy.
    """
    file_path = Path(file_path)
    valid_ext = ['.txt', '.md']
    if file_path.suffix.lower() == '.py':
        return chunk_python_files(content, max_chunk_size)
    elif file_path.suffix.lower() in valid_ext:
        return chunk_md_text(content, max_chunk_size)
    else:
        raise ValueError(
            f"Unsupported extension for this file path: {file_path}")
