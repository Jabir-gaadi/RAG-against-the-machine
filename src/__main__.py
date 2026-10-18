import fire
import sys
from .cli import (
    search_dataset, evaluate, index, search, answer, answer_dataset)


if __name__ == "__main__":
    try:
        fire.Fire({
            "search_dataset": search_dataset,
            "evaluate": evaluate,
            "index": index,
            "search": search,
            "answer": answer,
            "answer_dataset": answer_dataset
            })
    except (
        FileNotFoundError, PermissionError, ValueError, ZeroDivisionError
            ) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
