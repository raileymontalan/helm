import os
from typing import Any, Dict, Optional, cast
from threading import Lock
from helm.common.cache import CacheConfig
from helm.common.concurrency import ThreadSafeWrapper

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from helm.common.hierarchical_logger import htrack_block, hlog
from .caching_tokenizer import CachingTokenizer
from .tokenizer import cleanup_tokens


WrappedPreTrainedTokenizer = ThreadSafeWrapper[PreTrainedTokenizerBase]
"""Thread safe wrapper around AI Singapore's SEA-LION SEABPETokenizer.

Example usage:

    with wrapped_tokenizer as tokenizer:
        tokenizer.encode("...")
"""


class AISingaporeTokenizer(CachingTokenizer):
    _tokenizers: Dict[str, WrappedPreTrainedTokenizer] = {}
    _tokenizers_lock: Lock = Lock()

    def __init__(self, cache_config: CacheConfig, pretrained_model_name_or_path: Optional[str] = None, **kwargs):
        super().__init__(cache_config=cache_config)
        self._pretrained_model_name_or_path = pretrained_model_name_or_path
        self._kwargs = kwargs

    @staticmethod
    def create_tokenizer(pretrained_model_name_or_path: str, **kwargs) -> WrappedPreTrainedTokenizer:
        """Loads tokenizer using files from disk if they exist. Otherwise, downloads from HuggingFace."""
        # To avoid deadlocks when using HuggingFace tokenizers with multiple processes
        # TODO: Figure out if we actually need this.
        os.environ["TOKENIZERS_PARALLELISM"] = "False"

        try:
            return WrappedPreTrainedTokenizer(
                AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path, local_files_only=True, trust_remote_code=True, **kwargs
                )
            )
        except OSError:
            hlog(f"Local files do not exist for HuggingFace tokenizer: {pretrained_model_name_or_path}. Downloading...")
            return WrappedPreTrainedTokenizer(
                AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path, local_files_only=False, trust_remote_code=True, **kwargs
                )
            )

    @staticmethod
    def get_tokenizer(
        helm_tokenizer_name: str, pretrained_model_name_or_path: str, **kwargs
    ) -> WrappedPreTrainedTokenizer:
        """
        Checks if the desired tokenizer is cached. Creates the tokenizer if it's not cached.
        Returns the tokenizer.
        """
        with AISingaporeTokenizer._tokenizers_lock:
            if helm_tokenizer_name not in AISingaporeTokenizer._tokenizers:
                with htrack_block(
                    f"Loading {pretrained_model_name_or_path} (kwargs={kwargs}) "
                    f"for HELM tokenizer {helm_tokenizer_name} with Hugging Face Transformers"
                ):
                    # Keep the tokenizer in memory, so we don't recreate it for future requests
                    AISingaporeTokenizer._tokenizers[helm_tokenizer_name] = AISingaporeTokenizer.create_tokenizer(
                        pretrained_model_name_or_path, **kwargs
                    )
        return AISingaporeTokenizer._tokenizers[helm_tokenizer_name]

    def _get_tokenizer_for_request(self, request: Dict[str, Any]) -> WrappedPreTrainedTokenizer:
        """Method used in both _tokenize_do_it and _decode_do_it to get the tokenizer."""
        pretrained_model_name_or_path: str
        if self._pretrained_model_name_or_path:
            pretrained_model_name_or_path = self._pretrained_model_name_or_path
        else:
            pretrained_model_name_or_path = request["tokenizer"]
        return AISingaporeTokenizer.get_tokenizer(
            helm_tokenizer_name=request["tokenizer"],
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            **self._kwargs,
        )

    def _tokenize_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        if request["encode"]:
            if request["truncation"]:
                with self._get_tokenizer_for_request(request) as tokenizer:
                    tokens = tokenizer.encode(
                        request["text"],
                        truncation=request["truncation"],
                        max_length=request["max_length"],
                        add_special_tokens=False,
                    )
            else:
                with self._get_tokenizer_for_request(request) as tokenizer:
                    tokens = tokenizer.encode(request["text"], add_special_tokens=False)
        else:
            with self._get_tokenizer_for_request(request) as tokenizer:
                tokens = tokenizer.tokenize(request["text"])
            if tokens and type(tokens[0]) == bytes:
                tokens = [cast(bytes, token).decode(errors="ignore") for token in tokens]
            tokens = cleanup_tokens(tokens, request["tokenizer"])
        return {"tokens": tokens}

    def _decode_do_it(self, request: Dict[str, Any]) -> Dict[str, Any]:
        with self._get_tokenizer_for_request(request) as tokenizer:
            text = tokenizer.decode(
                request["tokens"], clean_up_tokenization_spaces=request["clean_up_tokenization_spaces"]
            )
        return {"text": text}