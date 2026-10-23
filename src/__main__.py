if __name__ == "__main__":
    try:
        import fire
        import sys
    except ImportError as e:
        print(f"Error: missing dependency: {e}", file=sys.stderr)
        SystemExit(1)
    except KeyboardInterrupt:
        print("Program terminated by user.")
        SystemExit(1)
    try:
        from .cli import (
            search_dataset, evaluate, index, search, answer, answer_dataset)
    except KeyboardInterrupt:
        print("Program terminated by user.")
        raise SystemExit(1)
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
        FileNotFoundError,
        PermissionError,
        ValueError,
        ZeroDivisionError,
        TypeError,
        ImportError
            ) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)
    except KeyboardInterrupt:
        print("Program terminated by user.")
        raise SystemExit(1)
