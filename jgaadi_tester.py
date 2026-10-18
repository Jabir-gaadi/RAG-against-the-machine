#!/usr/bin/env python3
import subprocess
RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'


def run_command(command):
    return subprocess.run(command, shell=True, capture_output=True, text=True)


def missing_raw_dir():
    result = run_command("uv run python -m src index --raw_data_path \
        no_such_directory --output_path /tmp/test-index.json")
    print("missing_raw_dir . . .")
    if result.stderr:
        print(f"{GREEN}VALID TEST:\n {RESET}{result.stderr}")
    else:
        print(f"{RED}Failed\n{RESET}{result.stdout}")


def invalid_chunk_size():
    result = run_command("uv run python -m src index --max_chunk_size 0")
    print("test with invalid chunk size . . .")
    if result.stderr:
        print(f"{GREEN}VALID\n{RESET} {result.stderr}")
    else:
        print(f"{RED}FAILED\n{RESET}{result.stdout}")


def check_chunk_size(chunk_size):
    print("Cheking the chunk size . . .")
    result = run_command("uv run python -c \"import json; \
        data=json.load(open(\'data/processed/index.json\')); print(len(data));\
            print(max(x[\'last_character_index\'] -\
                x[\'first_character_index\'] for x in data))\"")
    print(result.stdout)
    if str(chunk_size) in str(result.stdout):
        print(f"{GREEN}VALID CHUNK SIZE")
    else:
        print(f"{RED}FAILED{RESET}\n")


if __name__ == "__main__":
    missing_raw_dir()
    print("======================================")
    invalid_chunk_size()
    print("======================================")
    chunk_size = input("Enter the chunk size: ")
    check_chunk_size(chunk_size)
