from .models import MinimalSource
from pathlib import Path
import torch
from typing import Any
from transformers import AutoModelForCausalLM, AutoTokenizer


class Generator():
    """Generate grounded answers from retrieved source locations using Qwen."""

    def __init__(self, root_path: str | Path):
        """Initialize a lazy local Qwen generator.

        Args:
            root_path: Project root used to resolve corpus-relative paths.
        """
        self.root_path: Path = Path(root_path)
        self.model_name = "Qwen/Qwen3-0.6B"
        self.tokenizer: Any | None = None
        self.model: Any | None = None

    def load_context(self, min_sources: list[MinimalSource]) -> list[
            dict[str, list[int] | str]]:
        """Read the exact source ranges referenced by retrieved locations.

        Args:
            min_sources: Retrieved file paths and character ranges.

        Returns:
            Context records containing source metadata and chunk content.

        Raises:
            FileNotFoundError: If a referenced source file is missing.
            PermissionError: If a referenced source file cannot be read.
            ValueError: If a source range is outside its file content.
        """
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
            dict[str, list[int] | str]]) -> str:
        """Build a grounded RAG prompt from a query and source context.

        Args:
            query: Natural-language question to answer.
            context_record: Source metadata and text returned by
                ``load_context``.

        Returns:
            Prompt instructing Qwen to answer only from supplied evidence.
        """
        prompt = 'Answer only from the provided context.\n'
        prompt += 'Use only facts explicitly stated in the provided context.'
        prompt += 'If the context does not explicitly contain the answer, \
reply exactly:\n'
        prompt += '/"I do not have enough context to answer this question./"'
        prompt += 'Do not use prior knowledge, infer missing facts,'
        prompt += 'or expand abbreviations unless the context explicitly does \
so.'
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

    def load_model(self) -> None:
        """Lazily load the Qwen tokenizer and model for CPU inference."""
        if self.tokenizer is not None and self.model is not None:
            return
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float32
            )
        # self.model.to("cpu")
        self.model.eval()

    def generate_from_prompt(self, prompt: str) -> str:
        """Generate a deterministic Qwen response for a formatted prompt.

        Args:
            prompt: Grounded RAG prompt to send to the local model.

        Returns:
            Decoded generated answer without the original prompt tokens.

        Raises:
            RuntimeError: If the tokenizer or model fails to load.
        """
        self.load_model()
        if self.tokenizer is None or self.model is None:
            raise RuntimeError("error of loading the model in runtime")
        messages = [
            {"role": "user", "content": f"{prompt}"},
            ]
        model_input = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=False).to(self.model.device)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_input,
                max_new_tokens=256,
                do_sample=False
                )
        prompt_length = model_input["input_ids"].shape[1]
        decode_new_tok = self.tokenizer.decode(
            generated_ids[0, prompt_length:],
            skip_special_tokens=True
            )
        return str(decode_new_tok.strip())

    def answer_question(self, query: str, sources: list[MinimalSource]) -> str:
        """Generate a grounded answer from a query and retrieved sources.

        Args:
            query: Natural-language question to answer.
            sources: Retrieved locations used as evidence.

        Returns:
            A generated answer or an insufficient-context fallback message.
        """
        if not sources:
            return "I do not have enough context to answer this question."
        this_loaded_context = self.load_context(min_sources=sources)
        final_prompt = self.prompt_construction(query, this_loaded_context)
        return self.generate_from_prompt(final_prompt)
