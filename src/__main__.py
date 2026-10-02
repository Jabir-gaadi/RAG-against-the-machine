import fire
from .cli import search_dataset, evaluate, index


if __name__ == "__main__":
    fire.Fire({
        "search_dataset": search_dataset,
        "evaluate": evaluate,
        "index": index
        })
