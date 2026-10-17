index:
	uv run python3 -m src index --max_chunk_size 2000

search_dataset_docs:
	uv run python3 -m src search_dataset --dataset_path data/datasets/public/UnansweredQuestions/dataset_docs_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

search_dataset_code:
	uv run python3 -m src search_dataset --dataset_path data/datasets/public/UnansweredQuestions/dataset_code_public.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

moulinette_docs:
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_public.json data/datasets/public/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000

moulinette_code:
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_public.json data/datasets/public/AnsweredQuestions/dataset_code_public.json --k 10 --max_context_length 2000

code_eval:
	uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_code_public.json \
  --dataset_path data/datasets/public/AnsweredQuestions/dataset_code_public.json
docs_eval:
	uv run python -m src evaluate \
  --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
  --dataset_path data/datasets/public/AnsweredQuestions/dataset_docs_public.json

clean:
	find . -type d -name "__pycache__" -prune -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -prune -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -prune -exec rm -rf {} +

fclean: clean
	rm -rf data/processed data/output

pvsearch_dataset_docs:
	uv run python3 -m src search_dataset --dataset_path data/datasets/private/UnansweredQuestions/dataset_docs_private.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

pvsearch_dataset_code:
	uv run python3 -m src search_dataset --dataset_path data/datasets/private/UnansweredQuestions/dataset_code_private.json --k 10 --save_directory data/output/search_results/UnansweredQuestions

pvmoulinette_docs:
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_docs_private.json data/datasets/private/AnsweredQuestions/dataset_docs_private.json --k 10 --max_context_length 2000

pvmoulinette_code:
	./moulinette/moulinette-ubuntu evaluate_student_search_results data/output/search_results/UnansweredQuestions/dataset_code_private.json data/datasets/private/AnsweredQuestions/dataset_code_private.json --k 10 --max_context_length 2000