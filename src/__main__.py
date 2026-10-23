if __name__ == "__main__":
    try:
        # pyrefly: ignore[missing-import]
        import fire
        import sys
    except KeyboardInterrupt:
        print("Program terminated by user.")
    try:
        from .cli import (
            search_dataset, evaluate, index, search, answer, answer_dataset)
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
    except KeyboardInterrupt:
        print("Program terminated by user.")
        raise SystemExit(1)
