import fire
from .cli import (
    search_dataset, evaluate, index, search, answer, answer_dataset)


if __name__ == "__main__":
    fire.Fire({
        "search_dataset": search_dataset,
        "evaluate": evaluate,
        "index": index,
        "search": search,
        "answer": answer,
        "answer_dataset": answer_dataset
        })
