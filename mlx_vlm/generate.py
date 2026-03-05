import argparse
import codecs
import contextlib
import functools
import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_reduce
from mlx_lm.generate import maybe_quantize_kv_cache
from mlx_lm.sample_utils import make_logits_processors, make_sampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from .models import cache
from .prompt_utils import apply_chat_template
from .utils import StoppingCriteria, group_images_by_shape, load, prepare_inputs

DEFAULT_MODEL_PATH = "mlx-community/nanoLLaVA-1.5-8bit"
DEFAULT_IMAGE = None
DEFAULT_AUDIO = None
DEFAULT_PROMPT = "What are these?"
DEFAULT_MAX_TOKENS = 256
DEFAULT_TEMPERATURE = 0.5
DEFAULT_TOP_P = 1.0
DEFAULT_SEED = 0
DEFAULT_QUANTIZED_KV_START = 5000


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Generate text from an image using a model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="The path to the local model directory or Hugging Face repo.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="The path to the adapter weights.",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        default=DEFAULT_IMAGE,
        help="URL or path of the image to process.",
    )
    parser.add_argument(
        "--audio",
        type=str,
        nargs="+",
        default=DEFAULT_AUDIO,
        help="URL or path of the audio to process.",
    )
    parser.add_argument(
        "--resize-shape",
        type=int,
        nargs="+",
        default=None,
        help="Resize shape for the image.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=DEFAULT_PROMPT,
        help="Message to be processed by the model.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System message for the model.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Maximum number of tokens to generate.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Temperature for sampling.",
    )
    parser.add_argument("--chat", action="store_true", help="Chat in multi-turn style.")
    parser.add_argument("--verbose", action="store_false", help="Detailed output.")
    parser.add_argument(
        "--eos-tokens",
        type=str,
        nargs="+",
        default=None,
        help="EOS tokens to add to the tokenizer.",
    )
    parser.add_argument(
        "--max-kv-size",
        type=int,
        default=None,
        help="Maximum KV size for the prompt cache.",
    )
    parser.add_argument(
        "--kv-bits",
        type=int,
        default=None,
        help="Number of bits to quantize the KV cache to.",
    )
    parser.add_argument(
        "--kv-group-size",
        type=int,
        default=64,
        help="Group size for the KV cache.",
    )
    parser.add_argument(
        "--quantized-kv-start",
        type=int,
        default=DEFAULT_QUANTIZED_KV_START,
        help="Start index for the quantized KV cache.",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Skip special tokens in the detokenizer.",
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force download the model from Hugging Face.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The specific model version to use (branch, tag, commit).",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading the model.",
    )
    parser.add_argument(
        "--quantize-activations",
        "-qa",
        action="store_true",
        help="Enable activation quantization for QQLinear layers. "
        "Only supported for models quantized with 'nvfp4' or 'mxfp8' modes.",
    )
    parser.add_argument(
        "--processor-kwargs",
        type=json.loads,
        default={},
        help="Extra processor kwargs as JSON. "
        'Example: --processor-kwargs \'{"cropping": false, "max_patches": 3}\'',
    )
    parser.add_argument(
        "--prefill-step-size",
        type=int,
        default=None,
        help="Number of tokens to process per prefill step. "
        "Lower values reduce peak memory usage but may be slower. "
        "Try 512 or 256 if you hit GPU memory errors during prefill.",
    )
    parser.add_argument(
        "--early-exit-layer",
        type=int,
        default=None,
        help="Layer for self-speculative decoding (early exit). "
        "Uses the first N layers as a draft model for speculative decoding. "
        "Recommended: 25-50%% of total layers.",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Path to a draft VLM for speculative decoding. "
        "Must have the same tokenizer vocabulary as the main model.",
    )
    parser.add_argument(
        "--num-draft-tokens",
        type=int,
        default=3,
        help="Number of tokens to draft per speculative cycle. Default: 3.",
    )

    return parser.parse_args()


# A stream on the default device just for generation
generation_stream = mx.new_stream(mx.default_device())


@contextlib.contextmanager
def wired_limit(model: nn.Module, streams: Optional[List[mx.Stream]] = None):
    """
    A context manager to temporarily change the wired limit.

    Note, the wired limit should not be changed during an async eval.  If an
    async eval could be running pass in the streams to synchronize with prior
    to exiting the context manager.
    """
    if not mx.metal.is_available():
        yield
        return

    model_bytes = tree_reduce(
        lambda acc, x: acc + x.nbytes if isinstance(x, mx.array) else acc, model, 0
    )
    max_rec_size = mx.device_info()["max_recommended_working_set_size"]
    if model_bytes > 0.9 * max_rec_size:
        model_mb = model_bytes // 2**20
        max_rec_mb = max_rec_size // 2**20
        print(
            f"[WARNING] Generating with a model that requires {model_mb} MB "
            f"which is close to the maximum recommended size of {max_rec_mb} "
            "MB. This can be slow. See the documentation for possible work-arounds: "
            "https://github.com/ml-explore/mlx-lm/tree/main#large-models"
        )
    old_limit = mx.set_wired_limit(max_rec_size)
    try:
        yield
    finally:
        if streams is not None:
            for s in streams:
                mx.synchronize(s)
        else:
            mx.synchronize()
        mx.set_wired_limit(old_limit)


@dataclass
class GenerationResult:
    text: str = ""
    token: Optional[int] = None
    logprobs: Optional[List[float]] = None
    prompt_tokens: int = 0
    generation_tokens: int = 0
    total_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory: float = 0.0
    # Speculation metrics (only set when using speculative decoding)
    spec_accepted: int = 0  # draft tokens accepted
    spec_drafted: int = 0  # total draft tokens proposed
    spec_cycles: int = 0  # number of verify cycles


def generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values,
    mask,
    *,
    max_tokens: int = 256,
    temperature: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = 20,
    top_p: float = 1.0,
    logit_bias: Optional[Dict[int, float]] = None,
    prompt_cache: Optional[List[Any]] = None,
    max_kv_size: Optional[int] = None,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prefill_step_size: Optional[int] = 2048,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array], None, None]:
    """
    A generator producing token ids based on the given prompt from the model.

    Args:
        input_ids (mx.array): The input prompt token ids.
        model (nn.Module): The model to use for generation.
        pixel_values: The pixel values for vision models (optional).
        mask: The attention mask (optional).
        max_tokens (int): Maximum number of tokens to generate. Default: ``256``.
        temperature (float): The temperature for sampling, if 0 the argmax is used.
          Default: ``0``.
        repetition_penalty (float, optional): The penalty factor for repeating
          tokens.
        repetition_context_size (int, optional): The number of tokens to
          consider for repetition penalty. Default: ``20``.
        top_p (float, optional): Nucleus sampling, higher means model considers
          more less likely words.
        logit_bias (dictionary, optional): Additive logit bias.
        prompt_cache (list, optional): Pre-existing KV cache for the prompt.
        max_kv_size (int, optional): Maximum KV cache size.
        kv_bits (int, optional): Number of bits for KV cache quantization.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Start index for quantized KV cache. Default: ``0``.
        sampler (Callable[mx.array, mx.array], optional): A sampler for sampling a
          token from a vector of log probabilities. Default: ``None``.
        logits_processors (List[Callable[[mx.array, mx.array], mx.array]], optional):
          A list of functions that take tokens and logits and return the processed
          logits. Default: ``None``.
        prefill_step_size (int): Number of tokens to process per prefill step.
          Chunked prefill processes prompts in smaller chunks to reduce peak
          memory usage. Default: ``2048``.

    Yields:
        Generator[Tuple[mx.array, mx.array], None, None]: A generator producing
          one token and a vector of log probabilities.
    """

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    if sampler is None:
        sampler = make_sampler(temperature, top_p)

    processors = make_logits_processors(
        logit_bias, repetition_penalty, repetition_context_size
    )
    if logits_processors is not None:
        processors.extend(logits_processors)

    y = input_ids
    tokens = mx.array([], dtype=input_ids.dtype)

    # Create the KV cache for generation
    if prompt_cache is None:
        prompt_cache = cache.make_prompt_cache(
            model.language_model,
            max_kv_size=max_kv_size,
        )

    def _step(y, inputs_embeds=None):
        nonlocal tokens, kwargs

        with mx.stream(generation_stream):
            if "decoder_input_ids" in kwargs:
                outputs = model.language_model(
                    cache=prompt_cache,
                    **kwargs,
                )
            else:
                outputs = model.language_model(
                    y,
                    inputs_embeds=inputs_embeds,
                    cache=prompt_cache,
                    **kwargs,
                )

            logits = outputs.logits[:, -1, :]

            if len(processors) > 0 and len(y) > 0:
                tokens = mx.concat([tokens, y.flatten()])

                for processor in processors:
                    logits = processor(tokens, logits)

            quantize_cache_fn(prompt_cache)

            logprobs = logits - mx.logsumexp(logits)
            y = sampler(logprobs)

            if outputs.cross_attention_states is not None:
                kwargs = {"cross_attention_states": outputs.cross_attention_states}
            elif outputs.encoder_outputs is not None:
                kwargs = {"encoder_outputs": outputs.encoder_outputs}
            else:
                kwargs = {}

            return y, logprobs.squeeze(0)

    with mx.stream(generation_stream):

        # Get input embeddings (handles both multimodal and text-only)
        embedding_output = model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )

        inputs_embeds = embedding_output.inputs_embeds

        kwargs.update(
            {
                k: v
                for k, v in embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }
        )
        if prefill_step_size is not None and inputs_embeds.shape[1] > prefill_step_size:
            # Chunked prefill with embeddings
            total_tokens = inputs_embeds.shape[1]
            with tqdm(total=total_tokens, desc="Prefill", unit="tok") as pbar:
                while inputs_embeds.shape[1] > 1:
                    n_to_process = min(prefill_step_size, inputs_embeds.shape[1] - 1)
                    model.language_model(
                        inputs=input_ids[:, :n_to_process],
                        inputs_embeds=inputs_embeds[:, :n_to_process],
                        cache=prompt_cache,
                        n_to_process=n_to_process,
                        **kwargs,
                    )
                    quantize_cache_fn(prompt_cache)
                    mx.eval([c.state for c in prompt_cache])
                    inputs_embeds = inputs_embeds[:, n_to_process:]
                    input_ids = input_ids[:, n_to_process:]
                    mx.clear_cache()
                    pbar.update(n_to_process)

            input_ids = input_ids[:, -1:]

        y, logprobs = _step(input_ids, inputs_embeds=inputs_embeds)

    mx.async_eval(y)

    n = 0
    while True:
        if n != max_tokens:
            next_y, next_logprobs = _step(y[None])
            mx.async_eval(next_y)
        if n == 0:
            mx.eval(y)
        if n == max_tokens:
            break

        yield y.item(), logprobs
        if n % 256 == 0:
            mx.clear_cache()
        y, logprobs = next_y, next_logprobs
        n += 1


def self_speculative_generate_step(
    input_ids: mx.array,
    model: nn.Module,
    pixel_values,
    mask,
    *,
    early_exit_layer: int,
    num_draft_tokens: int = 3,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[List[Any]] = None,
    prefill_step_size: Optional[int] = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array, bool], None, None]:
    """
    A generator producing token ids using self-speculative decoding (early exit)
    for vision-language models.

    The draft phase runs only the first ``early_exit_layer`` layers of the
    language model, then the verify phase runs all layers. This eliminates the
    need for a separate draft model (LayerSkip-style self-speculative decoding).

    Args:
        input_ids (mx.array): The input prompt token ids.
        model (nn.Module): The vision-language model.
        pixel_values: The pixel values for vision models.
        mask: The attention mask.
        early_exit_layer (int): Number of language model layers for draft generation.
        num_draft_tokens (int): Number of draft tokens per cycle. Default: ``3``.
        max_tokens (int): Maximum number of tokens to generate. Default: ``256``.
        sampler: A sampler for sampling from log probabilities.
        logits_processors: Functions that process logits before sampling.
        prompt_cache: Pre-existing KV cache.
        prefill_step_size (int): Chunk size for prefill. Default: ``2048``.
        kv_bits (int, optional): Bits for KV cache quantization.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin quantized KV cache. Default: ``0``.

    Yields:
        Tuple[mx.array, mx.array, bool]: One token, log probabilities, and
          whether the token was generated by the draft (early exit).
    """

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    if sampler is None:
        sampler = make_sampler(0.0)  # greedy by default for speculative

    prev_tokens = None

    # Tier 1: two separate full caches (draft + verify)
    language_model = model.language_model
    if prompt_cache is not None:
        n_layers = len(language_model.layers)
        verify_cache = prompt_cache[:n_layers]
        draft_cache = prompt_cache[n_layers:]
    else:
        verify_cache = cache.make_prompt_cache(language_model)
        draft_cache = cache.make_prompt_cache(language_model)

    def _process_and_sample(tokens, logits):
        if logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        return y, logprobs

    def _lm_step(y, use_cache, n_predict=1, num_layers=None):
        nonlocal prev_tokens, step_kwargs
        with mx.stream(generation_stream):
            # Save position state before draft calls to prevent corruption
            if num_layers is not None:
                saved_pos = (
                    getattr(language_model, "_position_ids", None),
                    getattr(language_model, "_rope_deltas", None),
                )

            outputs = language_model(
                y,
                cache=use_cache,
                num_layers=num_layers,
                **step_kwargs,
            )
            logits = outputs.logits[:, -n_predict:, :]
            quantize_cache_fn(use_cache)

            # Restore position state after draft calls
            if num_layers is not None:
                language_model._position_ids, language_model._rope_deltas = saved_pos
            else:
                # Update step_kwargs only from verify outputs (not draft)
                if outputs.cross_attention_states is not None:
                    step_kwargs = {
                        "cross_attention_states": outputs.cross_attention_states
                    }
                elif outputs.encoder_outputs is not None:
                    step_kwargs = {"encoder_outputs": outputs.encoder_outputs}

            if logits_processors and n_predict > 1:
                out_y, out_logprobs = [], []
                flat_y = y.flatten()[: -(n_predict - 1)] if n_predict > 1 else y.flatten()
                for i in range(n_predict):
                    prev_tokens = (
                        mx.concatenate([prev_tokens, flat_y])
                        if prev_tokens is not None
                        else flat_y
                    )
                    yi, lpi = _process_and_sample(prev_tokens, logits[:, i, :])
                    out_y.append(yi)
                    out_logprobs.append(lpi)
                    flat_y = yi
                return mx.concatenate(out_y, axis=0), mx.concatenate(out_logprobs, axis=0)
            else:
                return _process_and_sample(None, logits.squeeze(0))

    # Check if caches have non-trimmable entries (hybrid models like Qwen3.5
    # with ArraysCache for linear attention). These need save/restore for rewind.
    _has_non_trimmable = not all(c.is_trimmable() for c in verify_cache)

    def _rewind_cache(num_draft, num_accept):
        """Trim trimmable cache entries (KVCache). Non-trimmable entries
        are handled by save/restore when _has_non_trimmable is True."""
        n_trim_verify = num_draft - num_accept
        n_trim_draft = max(num_draft - num_accept - 1, 0)
        if n_trim_verify > 0:
            for c in verify_cache:
                if c.is_trimmable():
                    c.trim(n_trim_verify)
        if n_trim_draft > 0:
            for c in draft_cache:
                if c.is_trimmable():
                    c.trim(n_trim_draft)

    def _save_cache_state(cache_list):
        """Save full cache state for rollback (both trimmable and non-trimmable).

        For non-trimmable entries (ArraysCache): copies arrays eagerly.
        For trimmable entries (KVCache): saves offset for trim-based restore.
        """
        saved = {}
        arrays_to_eval = []
        for i, c in enumerate(cache_list):
            if not c.is_trimmable() and hasattr(c, "state"):
                state = c.state
                if state is not None and isinstance(state, list):
                    copies = []
                    for s in state:
                        if isinstance(s, mx.array):
                            copy = mx.array(s)
                            copies.append(copy)
                            arrays_to_eval.append(copy)
                        else:
                            copies.append(s)
                    saved[i] = ("arrays", copies)
        if arrays_to_eval:
            mx.eval(arrays_to_eval)
        return saved

    def _restore_cache_state(cache_list, saved):
        """Restore saved cache state."""
        for i, (kind, state) in saved.items():
            if kind == "arrays":
                cache_list[i].state = state

    def _draft_generate(y, num_draft):
        if num_draft == 0:
            return mx.array([], mx.uint32)
        ys = []
        for _ in range(num_draft):
            y_out, _ = _lm_step(y[None], draft_cache, num_layers=early_exit_layer)
            mx.async_eval(y_out)
            ys.append(y_out)
            y = y_out
        return mx.concatenate(ys)

    # --- Prefill with vision embeddings ---
    step_kwargs = {}

    with mx.stream(generation_stream):
        # Get input embeddings (handles multimodal fusion)
        embedding_output = model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        inputs_embeds = embedding_output.inputs_embeds

        step_kwargs.update(
            {
                k: v
                for k, v in embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }
        )

        # Prefill verify cache with full model
        if prefill_step_size is not None and inputs_embeds.shape[1] > prefill_step_size:
            remaining = inputs_embeds
            remaining_ids = input_ids
            while remaining.shape[1] > 1:
                n_to_process = min(prefill_step_size, remaining.shape[1] - 1)
                language_model(
                    remaining_ids[:, :n_to_process],
                    inputs_embeds=remaining[:, :n_to_process],
                    cache=verify_cache,
                    **step_kwargs,
                )
                quantize_cache_fn(verify_cache)
                mx.eval([c.state for c in verify_cache])
                remaining = remaining[:, n_to_process:]
                remaining_ids = remaining_ids[:, n_to_process:]
                mx.clear_cache()
            # Final token
            outputs = language_model(
                remaining_ids,
                inputs_embeds=remaining,
                cache=verify_cache,
                **step_kwargs,
            )
        else:
            outputs = language_model(
                input_ids,
                inputs_embeds=inputs_embeds,
                cache=verify_cache,
                **step_kwargs,
            )

        # Handle cross-attention/encoder outputs from prefill
        if outputs.cross_attention_states is not None:
            step_kwargs = {"cross_attention_states": outputs.cross_attention_states}
        elif outputs.encoder_outputs is not None:
            step_kwargs = {"encoder_outputs": outputs.encoder_outputs}
        else:
            step_kwargs = {}

        quantize_cache_fn(verify_cache)

        # Force evaluation of verify outputs before draft prefill,
        # to prevent MLX lazy evaluation from interleaving computations
        # that share the same model weights
        mx.eval(outputs.logits)
        mx.eval([c.state for c in verify_cache])

        # Prefill draft cache with early exit layers
        # Save position state set by verify prefill; draft prefill will overwrite it
        saved_pos = (
            getattr(language_model, "_position_ids", None),
            getattr(language_model, "_rope_deltas", None),
        )
        if prefill_step_size is not None and inputs_embeds.shape[1] > prefill_step_size:
            remaining = inputs_embeds
            remaining_ids = input_ids
            while remaining.shape[1] > 1:
                n_to_process = min(prefill_step_size, remaining.shape[1] - 1)
                language_model(
                    remaining_ids[:, :n_to_process],
                    inputs_embeds=remaining[:, :n_to_process],
                    cache=draft_cache,
                    num_layers=early_exit_layer,
                    **step_kwargs,
                )
                quantize_cache_fn(draft_cache)
                mx.eval([c.state for c in draft_cache])
                remaining = remaining[:, n_to_process:]
                remaining_ids = remaining_ids[:, n_to_process:]
                mx.clear_cache()
            language_model(
                remaining_ids,
                inputs_embeds=remaining,
                cache=draft_cache,
                num_layers=early_exit_layer,
                **step_kwargs,
            )
        else:
            language_model(
                input_ids,
                inputs_embeds=inputs_embeds,
                cache=draft_cache,
                num_layers=early_exit_layer,
                **step_kwargs,
            )
        quantize_cache_fn(draft_cache)
        # Restore position state for verify path
        language_model._position_ids, language_model._rope_deltas = saved_pos

    # Sample first token from verify logits
    logits = outputs.logits[:, -1:, :]
    y, logprobs_first = _process_and_sample(None, logits.squeeze(0))
    mx.eval(y)

    # Yield the first generated token
    yield y.item(), logprobs_first, False

    ntoks = 1
    num_draft = 0
    n = 0
    draft_y = y

    # Adaptive draft count: track recent acceptance to avoid wasted drafts.
    # When acceptance is consistently 0%, drafting costs N extra model calls
    # for no benefit, so we temporarily reduce draft count to 0 (= baseline).
    _accept_history = 0  # bitmask of last 8 cycles (1 = any acceptance)
    _draft_cycle_count = 0
    _baseline_count = 0
    _probe_interval = 32  # tokens between probe attempts (doubles on failure)
    _adaptive_draft = num_draft_tokens  # current effective draft count

    try:
        while True:
            num_draft = min(max_tokens - ntoks, _adaptive_draft)

            if num_draft > 0 and _has_non_trimmable:
                _verify_saved = _save_cache_state(verify_cache)
                _draft_saved = _save_cache_state(draft_cache)

            draft_tokens = _draft_generate(draft_y, num_draft)

            if num_draft == 0:
                # No drafting — just run verify for one token (baseline speed)
                verify_tok, verify_lp = _lm_step(y[None], verify_cache)
                mx.eval(verify_tok)
                n = 0
                ntoks += 1
                yield verify_tok.item(), verify_lp, False
                y = verify_tok
                draft_y = y
            else:
                # Batch verify: process all draft tokens at once
                if prev_tokens is not None:
                    prev_tokens = prev_tokens[
                        : prev_tokens.size - y.size - num_draft + 1
                    ]
                all_tokens = mx.concatenate([y, draft_tokens])
                tokens, logprobs = _lm_step(
                    all_tokens[None], verify_cache, num_draft + 1
                )
                mx.eval(tokens, draft_tokens)
                y_prev_item = y.item()
                draft_tokens = draft_tokens.tolist()
                tokens = tokens.tolist()
                n = 0
                while n < num_draft:
                    tn, dtn, lpn = tokens[n], draft_tokens[n], logprobs[n]
                    if tn != dtn:
                        break
                    n += 1
                    ntoks += 1
                    yield tn, lpn, True
                    if ntoks == max_tokens:
                        break
                if ntoks < max_tokens:
                    ntoks += 1
                    yield tokens[n], logprobs[n], False

                if ntoks < max_tokens:
                    y = mx.array([tokens[n]], mx.uint32)
                    draft_y = y
                    if n == num_draft:
                        draft_y = mx.concatenate(
                            [mx.array(draft_tokens[-1:], mx.uint32), draft_y]
                        )
                    if prev_tokens is not None:
                        prev_tokens = prev_tokens[: -max(num_draft - n, 1)]

                    if _has_non_trimmable:
                        if n == num_draft:
                            # All drafts accepted. Draft cache hasn't seen
                            # the last draft token (it was output, not input).
                            # Feed it now to keep ArraysCache in sync with KV.
                            catchup = mx.array(draft_tokens[-1:], mx.uint32)
                            _lm_step(
                                catchup[None], draft_cache,
                                num_layers=early_exit_layer,
                            )
                            draft_y = y  # single token, not [last_draft, bonus]
                        else:
                            # Partial acceptance — restore state, replay
                            _restore_cache_state(verify_cache, _verify_saved)
                            _restore_cache_state(draft_cache, _draft_saved)
                            for c in verify_cache:
                                if c.is_trimmable():
                                    c.trim(num_draft + 1)
                            for c in draft_cache:
                                if c.is_trimmable():
                                    c.trim(num_draft)
                            replay_ids = [y_prev_item] + draft_tokens[:n]
                            replay = mx.array([replay_ids], mx.uint32)
                            _lm_step(replay, verify_cache, len(replay_ids))
                            _lm_step(
                                replay, draft_cache, len(replay_ids),
                                num_layers=early_exit_layer,
                            )
                    else:
                        _rewind_cache(num_draft, n)

            if ntoks == max_tokens:
                break

            # Update adaptive draft count based on acceptance history
            if num_draft > 0:
                _accept_history = ((_accept_history << 1) | (1 if n > 0 else 0)) & 0xFF
                _draft_cycle_count = min(_draft_cycle_count + 1, 8)
                if _draft_cycle_count >= 2 and _accept_history == 0:
                    _adaptive_draft = 0
                    _baseline_count = 0
                elif n > 0:
                    _probe_interval = 32
            else:
                _baseline_count += 1
                if _baseline_count >= _probe_interval:
                    _adaptive_draft = 1
                    _draft_cycle_count = 0
                    _accept_history = 0
                    _baseline_count = 0
                    _probe_interval = min(_probe_interval * 2, 256)
    finally:
        if num_draft > 0:
            if _has_non_trimmable:
                _restore_cache_state(verify_cache, _verify_saved)
                for c in verify_cache:
                    if c.is_trimmable():
                        c.trim(num_draft + 1)
            else:
                _rewind_cache(num_draft, n)


def speculative_generate_step(
    input_ids: mx.array,
    model: nn.Module,
    draft_model: nn.Module,
    pixel_values,
    mask,
    *,
    num_draft_tokens: int = 3,
    max_tokens: int = 256,
    sampler: Optional[Callable[[mx.array], mx.array]] = None,
    logits_processors: Optional[List[Callable[[mx.array, mx.array], mx.array]]] = None,
    prompt_cache: Optional[List[Any]] = None,
    prefill_step_size: Optional[int] = 2048,
    kv_bits: Optional[int] = None,
    kv_group_size: int = 64,
    quantized_kv_start: int = 0,
    **kwargs,
) -> Generator[Tuple[mx.array, mx.array, bool], None, None]:
    """
    A generator producing token ids using speculative decoding with a separate
    draft vision-language model.

    Args:
        input_ids (mx.array): The input prompt token ids.
        model (nn.Module): The full VLM (verifier).
        draft_model (nn.Module): The smaller VLM (drafter).
        pixel_values: The pixel values for vision models.
        mask: The attention mask.
        num_draft_tokens (int): Number of draft tokens per cycle. Default: ``3``.
        max_tokens (int): Maximum number of tokens to generate. Default: ``256``.
        sampler: A sampler for sampling from log probabilities.
        logits_processors: Functions that process logits before sampling.
        prompt_cache: Pre-existing KV cache.
        prefill_step_size (int): Chunk size for prefill. Default: ``2048``.
        kv_bits (int, optional): Bits for KV cache quantization.
        kv_group_size (int): Group size for KV cache quantization. Default: ``64``.
        quantized_kv_start (int): Step to begin quantized KV cache. Default: ``0``.

    Yields:
        Tuple[mx.array, mx.array, bool]: One token, log probabilities, and
          whether the token was generated by the draft model.
    """

    quantize_cache_fn = functools.partial(
        maybe_quantize_kv_cache,
        quantized_kv_start=quantized_kv_start,
        kv_group_size=kv_group_size,
        kv_bits=kv_bits,
    )

    if sampler is None:
        sampler = make_sampler(0.0)

    prev_tokens = None

    language_model = model.language_model
    draft_language_model = draft_model.language_model

    if prompt_cache is not None:
        n_model_layers = len(language_model.layers)
        model_cache = prompt_cache[:n_model_layers]
        draft_cache = prompt_cache[n_model_layers:]
    else:
        model_cache = cache.make_prompt_cache(language_model)
        draft_cache = cache.make_prompt_cache(draft_language_model)

    def _process_and_sample(tokens, logits):
        if logits_processors:
            for processor in logits_processors:
                logits = processor(tokens, logits)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        y = sampler(logprobs)
        return y, logprobs

    def _step(lm, lm_cache, y, step_kw, n_predict=1):
        nonlocal prev_tokens
        with mx.stream(generation_stream):
            outputs = lm(y, cache=lm_cache, **step_kw)
            logits = outputs.logits[:, -n_predict:, :]
            quantize_cache_fn(lm_cache)

            if logits_processors and n_predict > 1:
                out_y, out_logprobs = [], []
                flat_y = (
                    y.flatten()[: -(n_predict - 1)] if n_predict > 1 else y.flatten()
                )
                for i in range(n_predict):
                    prev_tokens = (
                        mx.concatenate([prev_tokens, flat_y])
                        if prev_tokens is not None
                        else flat_y
                    )
                    yi, lpi = _process_and_sample(prev_tokens, logits[:, i, :])
                    out_y.append(yi)
                    out_logprobs.append(lpi)
                    flat_y = yi
                return mx.concatenate(out_y, axis=0), mx.concatenate(
                    out_logprobs, axis=0
                ), outputs
            else:
                y_out, lp_out = _process_and_sample(None, logits.squeeze(0))
                return y_out, lp_out, outputs

    def _prefill_lm(lm, lm_cache, inputs_embeds, ids, step_kw):
        """Prefill a language model cache, returns final outputs."""
        if prefill_step_size is not None and inputs_embeds.shape[1] > prefill_step_size:
            remaining = inputs_embeds
            remaining_ids = ids
            while remaining.shape[1] > 1:
                n_to_process = min(prefill_step_size, remaining.shape[1] - 1)
                lm(
                    remaining_ids[:, :n_to_process],
                    inputs_embeds=remaining[:, :n_to_process],
                    cache=lm_cache,
                    **step_kw,
                )
                quantize_cache_fn(lm_cache)
                mx.eval([c.state for c in lm_cache])
                remaining = remaining[:, n_to_process:]
                remaining_ids = remaining_ids[:, n_to_process:]
                mx.clear_cache()
            outputs = lm(
                remaining_ids,
                inputs_embeds=remaining,
                cache=lm_cache,
                **step_kw,
            )
        else:
            outputs = lm(
                ids,
                inputs_embeds=inputs_embeds,
                cache=lm_cache,
                **step_kw,
            )
        quantize_cache_fn(lm_cache)
        return outputs

    def _rewind_cache(num_draft, num_accept):
        cache.trim_prompt_cache(model_cache, num_draft - num_accept)
        cache.trim_prompt_cache(draft_cache, max(num_draft - num_accept - 1, 0))

    def _draft_generate(y, num_draft):
        if num_draft == 0:
            return mx.array([], mx.uint32)
        ys = []
        for _ in range(num_draft):
            y_out, _, _ = _step(
                draft_language_model, draft_cache, y[None], draft_step_kwargs
            )
            mx.async_eval(y_out)
            ys.append(y_out)
            y = y_out
        return mx.concatenate(ys)

    # --- Prefill both models with vision embeddings ---
    model_step_kwargs = {}
    draft_step_kwargs = {}

    with mx.stream(generation_stream):
        # Get embeddings from verify model
        embedding_output = model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        model_inputs_embeds = embedding_output.inputs_embeds
        model_step_kwargs.update(
            {
                k: v
                for k, v in embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }
        )

        # Get embeddings from draft model
        draft_embedding_output = draft_model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **kwargs
        )
        draft_inputs_embeds = draft_embedding_output.inputs_embeds
        draft_step_kwargs.update(
            {
                k: v
                for k, v in draft_embedding_output.to_dict().items()
                if k != "inputs_embeds" and v is not None
            }
        )

        # Prefill verify model
        outputs = _prefill_lm(
            language_model, model_cache, model_inputs_embeds, input_ids,
            model_step_kwargs,
        )
        if outputs.cross_attention_states is not None:
            model_step_kwargs = {
                "cross_attention_states": outputs.cross_attention_states
            }
        elif outputs.encoder_outputs is not None:
            model_step_kwargs = {"encoder_outputs": outputs.encoder_outputs}
        else:
            model_step_kwargs = {}

        # Prefill draft model
        draft_outputs = _prefill_lm(
            draft_language_model, draft_cache, draft_inputs_embeds, input_ids,
            draft_step_kwargs,
        )
        if draft_outputs.cross_attention_states is not None:
            draft_step_kwargs = {
                "cross_attention_states": draft_outputs.cross_attention_states
            }
        elif draft_outputs.encoder_outputs is not None:
            draft_step_kwargs = {"encoder_outputs": draft_outputs.encoder_outputs}
        else:
            draft_step_kwargs = {}

    # Sample first token from verify logits
    logits = outputs.logits[:, -1:, :]
    y, logprobs_first = _process_and_sample(None, logits.squeeze(0))
    mx.eval(y)

    # Yield the first generated token
    yield y.item(), logprobs_first, False

    # Check if caches have non-trimmable entries (hybrid models)
    _has_non_trimmable = not all(c.is_trimmable() for c in model_cache)

    def _rewind_cache(num_draft, num_accept):
        n_trim_verify = num_draft - num_accept
        n_trim_draft = max(num_draft - num_accept - 1, 0)
        if n_trim_verify > 0:
            for c in model_cache:
                if c.is_trimmable():
                    c.trim(n_trim_verify)
        if n_trim_draft > 0:
            for c in draft_cache:
                if c.is_trimmable():
                    c.trim(n_trim_draft)

    def _save_cache_state(cache_list):
        saved = {}
        arrays_to_eval = []
        for i, c in enumerate(cache_list):
            if not c.is_trimmable() and hasattr(c, "state"):
                state = c.state
                if state is not None and isinstance(state, list):
                    copies = []
                    for s in state:
                        if isinstance(s, mx.array):
                            copy = mx.array(s)
                            copies.append(copy)
                            arrays_to_eval.append(copy)
                        else:
                            copies.append(s)
                    saved[i] = ("arrays", copies)
        if arrays_to_eval:
            mx.eval(arrays_to_eval)
        return saved

    def _restore_cache_state(cache_list, saved):
        for i, (kind, state) in saved.items():
            if kind == "arrays":
                cache_list[i].state = state

    def _update_step_kwargs(outputs):
        nonlocal model_step_kwargs
        if outputs.cross_attention_states is not None:
            model_step_kwargs = {
                "cross_attention_states": outputs.cross_attention_states
            }
        elif outputs.encoder_outputs is not None:
            model_step_kwargs = {"encoder_outputs": outputs.encoder_outputs}

    ntoks = 1
    num_draft = 0
    n = 0
    draft_y = y

    # Adaptive draft count: track recent acceptance to avoid wasted drafts.
    _accept_history = 0
    _draft_cycle_count = 0
    _baseline_count = 0
    _probe_interval = 32
    _adaptive_draft = num_draft_tokens

    import os
    _spec_debug = os.environ.get("SPEC_DEBUG", "")

    try:
        while True:
            num_draft = min(max_tokens - ntoks, _adaptive_draft)

            if num_draft > 0 and _has_non_trimmable:
                _verify_saved = _save_cache_state(model_cache)
                _draft_saved = _save_cache_state(draft_cache)

            draft_tokens = _draft_generate(draft_y, num_draft)

            if num_draft == 0:
                # No drafting — just run verify for one token (baseline speed)
                verify_tok, verify_lp, verify_outputs = _step(
                    language_model, model_cache, y[None],
                    model_step_kwargs, 1,
                )
                _update_step_kwargs(verify_outputs)
                mx.eval(verify_tok)
                n = 0
                ntoks += 1
                yield verify_tok.item(), verify_lp, False
                y = verify_tok
                draft_y = y
            else:
                # Batch verify: process all draft tokens at once
                if prev_tokens is not None:
                    prev_tokens = prev_tokens[
                        : prev_tokens.size - y.size - num_draft + 1
                    ]
                all_tokens = mx.concatenate([y, draft_tokens])
                tokens, logprobs, verify_outputs = _step(
                    language_model, model_cache, all_tokens[None],
                    model_step_kwargs, num_draft + 1,
                )
                _update_step_kwargs(verify_outputs)

                mx.eval(tokens, draft_tokens)
                y_prev_item = y.item()
                draft_tokens = draft_tokens.tolist()
                tokens = tokens.tolist()
                n = 0
                while n < num_draft:
                    tn, dtn, lpn = tokens[n], draft_tokens[n], logprobs[n]
                    if tn != dtn:
                        break
                    n += 1
                    ntoks += 1
                    yield tn, lpn, True
                    if ntoks == max_tokens:
                        break
                if ntoks < max_tokens:
                    ntoks += 1
                    yield tokens[n], logprobs[n], False

                if _spec_debug:
                    _fa_idx = getattr(language_model, 'model', None)
                    _fa_idx = getattr(_fa_idx, 'fa_idx', None) if _fa_idx else None
                    _v_off = model_cache[_fa_idx].offset if _fa_idx is not None else '?'
                    _d_fa = getattr(draft_language_model, 'model', None)
                    _d_fa = getattr(_d_fa, 'fa_idx', None) if _d_fa else None
                    _d_off = draft_cache[_d_fa].offset if _d_fa is not None else '?'
                    print(f"  [SPEC] ntoks={ntoks} n={n}/{num_draft} "
                          f"v_off={_v_off} d_off={_d_off} "
                          f"draft_y_len={draft_y.size} "
                          f"y_prev={y_prev_item} "
                          f"tokens={tokens[:n+1]} draft={draft_tokens[:n+1]}")

                if ntoks < max_tokens:
                    y = mx.array([tokens[n]], mx.uint32)
                    draft_y = y
                    if n == num_draft:
                        draft_y = mx.concatenate(
                            [mx.array(draft_tokens[-1:], mx.uint32), draft_y]
                        )
                    if prev_tokens is not None:
                        prev_tokens = prev_tokens[: -max(num_draft - n, 1)]

                    if _has_non_trimmable:
                        if n == num_draft:
                            # All drafts accepted. Draft cache hasn't seen
                            # the last draft token. Feed it to sync.
                            catchup = mx.array(draft_tokens[-1:], mx.uint32)
                            _step(
                                draft_language_model, draft_cache,
                                catchup[None], draft_step_kwargs,
                            )
                            draft_y = y  # single token, not [last_draft, bonus]
                            if _spec_debug:
                                print(f"    [CATCHUP] fed {draft_tokens[-1]} to draft")
                        else:
                            _restore_cache_state(model_cache, _verify_saved)
                            _restore_cache_state(draft_cache, _draft_saved)
                            for c in model_cache:
                                if c.is_trimmable():
                                    c.trim(num_draft + 1)
                            for c in draft_cache:
                                if c.is_trimmable():
                                    c.trim(num_draft)
                            replay_ids = [y_prev_item] + draft_tokens[:n]
                            replay = mx.array([replay_ids], mx.uint32)
                            _step(language_model, model_cache, replay,
                                  model_step_kwargs, len(replay_ids))
                            _step(draft_language_model, draft_cache, replay,
                                  draft_step_kwargs, len(replay_ids))
                            if _spec_debug:
                                _v_off2 = model_cache[_fa_idx].offset if _fa_idx is not None else '?'
                                _d_off2 = draft_cache[_d_fa].offset if _d_fa is not None else '?'
                                print(f"    [REPLAY] {replay_ids} v_off={_v_off2} d_off={_d_off2}")
                    else:
                        _rewind_cache(num_draft, n)

            if ntoks == max_tokens:
                break

            # Update adaptive draft count based on acceptance history
            if num_draft > 0:
                _accept_history = ((_accept_history << 1) | (1 if n > 0 else 0)) & 0xFF
                _draft_cycle_count = min(_draft_cycle_count + 1, 8)
                if _draft_cycle_count >= 2 and _accept_history == 0:
                    _adaptive_draft = 0
                    _baseline_count = 0
                elif n > 0:
                    _probe_interval = 32
            else:
                _baseline_count += 1
                if _baseline_count >= _probe_interval:
                    _adaptive_draft = 1
                    _draft_cycle_count = 0
                    _accept_history = 0
                    _baseline_count = 0
                    _probe_interval = min(_probe_interval * 2, 256)
    finally:
        if num_draft > 0:
            if _has_non_trimmable:
                _restore_cache_state(model_cache, _verify_saved)
                for c in model_cache:
                    if c.is_trimmable():
                        c.trim(num_draft + 1)
            else:
                _rewind_cache(num_draft, n)


def stream_generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    audio: Union[str, List[str]] = None,
    **kwargs,
) -> Union[str, Generator[str, None, None]]:
    """
    A generator producing text based on the given prompt from the model.

    Args:
        model (nn.Module): The model to use for generation.
        processor (PreTrainedTokenizer): The tokenizer/processor.
        prompt (str): The input prompt text.
        image (Union[str, List[str]], optional): Image path(s) or URL(s).
        audio (Union[str, List[str]], optional): Audio file path(s).
        prefill_step_size (int, optional): Number of tokens to process per prefill
          step. When set, enables chunked prefill which processes long prompts in
          smaller chunks to reduce peak memory usage.
        kwargs: Additional options passed to :func:`generate_step`.
          See :func:`generate_step` for more details.

    Yields:
        Generator[GenerationResult]: A generator producing GenerationResult objects
          containing the generated text, tokens, and statistics.
    """
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Skip special tokens
    skip_special_tokens = kwargs.pop("skip_special_tokens", False)
    skip_special_token_ids = (
        set(tokenizer.all_special_ids)
        if skip_special_tokens and hasattr(tokenizer, "all_special_ids")
        else []
    )

    add_special_tokens = (
        not hasattr(processor, "chat_template")
        if model.config.model_type in ["gemma3", "gemma3n"]
        else True
    )

    resize_shape = kwargs.pop("resize_shape", None)
    image_token_index = getattr(model.config, "image_token_index", None)

    if kwargs.get("input_ids", None) is not None:
        input_ids = kwargs.pop("input_ids")
        pixel_values = kwargs.pop("pixel_values", None)
        mask = kwargs.pop("mask", None)
    else:
        inputs = prepare_inputs(
            processor,
            images=image,
            audio=audio,
            prompts=prompt,
            image_token_index=image_token_index,
            resize_shape=resize_shape,
            add_special_tokens=add_special_tokens,
            **kwargs,
        )
        input_ids = inputs.get("input_ids", None)
        pixel_values = inputs.get("pixel_values", None)
        mask = inputs.get("attention_mask", None)
        data_kwargs = {
            k: v
            for k, v in inputs.items()
            if k not in ["input_ids", "pixel_values", "attention_mask"]
        }
        kwargs.update(data_kwargs)

    early_exit_layer = kwargs.pop("early_exit_layer", None)
    draft_model = kwargs.pop("draft_model", None)

    if early_exit_layer is not None and draft_model is not None:
        raise ValueError("Cannot use both --early-exit-layer and --draft-model")

    with wired_limit(model, [generation_stream]):
        detokenizer = processor.detokenizer
        detokenizer.reset()
        tic = time.perf_counter()

        if early_exit_layer is not None or draft_model is not None:
            if early_exit_layer is not None:
                token_generator = self_speculative_generate_step(
                    input_ids, model, pixel_values, mask,
                    early_exit_layer=early_exit_layer, **kwargs
                )
            else:
                token_generator = speculative_generate_step(
                    input_ids, model, draft_model, pixel_values, mask, **kwargs
                )
            spec_accepted = 0
            spec_cycles = 0
            for n, (token, logprobs, from_draft) in enumerate(token_generator):
                if n == 0:
                    prompt_time = time.perf_counter() - tic
                    prompt_tps = input_ids.size / prompt_time
                    tic = time.perf_counter()

                # Track speculation stats:
                # from_draft=True means an accepted draft token
                # from_draft=False means a verify token (ends a cycle)
                if from_draft:
                    spec_accepted += 1
                else:
                    spec_cycles += 1

                if tokenizer.stopping_criteria(token):
                    break

                detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)
                yield GenerationResult(
                    text=detokenizer.last_segment,
                    token=token,
                    logprobs=logprobs,
                    prompt_tokens=input_ids.size,
                    generation_tokens=n + 1,
                    total_tokens=input_ids.size + n + 1,
                    prompt_tps=prompt_tps,
                    generation_tps=(n + 1) / (time.perf_counter() - tic),
                    peak_memory=mx.get_peak_memory() / 1e9,
                    spec_accepted=spec_accepted,
                    spec_drafted=spec_accepted + spec_cycles,
                    spec_cycles=spec_cycles,
                )
        else:
            for n, (token, logprobs) in enumerate(
                generate_step(input_ids, model, pixel_values, mask, **kwargs)
            ):
                if n == 0:
                    prompt_time = time.perf_counter() - tic
                    prompt_tps = input_ids.size / prompt_time
                    tic = time.perf_counter()

                if tokenizer.stopping_criteria(token):
                    break

                detokenizer.add_token(token, skip_special_token_ids=skip_special_token_ids)
                yield GenerationResult(
                    text=detokenizer.last_segment,
                    token=token,
                    logprobs=logprobs,
                    prompt_tokens=input_ids.size,
                    generation_tokens=n + 1,
                    total_tokens=input_ids.size + n + 1,
                    prompt_tps=prompt_tps,
                    generation_tps=(n + 1) / (time.perf_counter() - tic),
                    peak_memory=mx.get_peak_memory() / 1e9,
                )

        detokenizer.finalize()
        spec_stats = {}
        if early_exit_layer is not None or draft_model is not None:
            spec_stats = dict(
                spec_accepted=spec_accepted,
                spec_drafted=spec_accepted + spec_cycles,
                spec_cycles=spec_cycles,
            )
        yield GenerationResult(
            text=detokenizer.last_segment,
            token=token,
            logprobs=logprobs,
            prompt_tokens=input_ids.size,
            generation_tokens=n + 1,
            total_tokens=input_ids.size + n + 1,
            prompt_tps=prompt_tps,
            generation_tps=(n + 1) / (time.perf_counter() - tic),
            peak_memory=mx.get_peak_memory() / 1e9,
            **spec_stats,
        )

        # Cleanup after generation
        mx.clear_cache()


def generate(
    model: nn.Module,
    processor: PreTrainedTokenizer,
    prompt: str,
    image: Union[str, List[str]] = None,
    audio: Union[str, List[str]] = None,
    verbose: bool = False,
    **kwargs,
) -> GenerationResult:
    """
    Generate text from the model.

    Args:
       model (nn.Module): The language model.
       tokenizer (PreTrainedTokenizer): The tokenizer.
       prompt (str): The string prompt.
       temperature (float): The temperature for sampling (default 0).
       max_tokens (int): The maximum number of tokens (default 100).
       verbose (bool): If ``True``, print tokens and timing information
           (default ``False``).
       formatter (Optional[Callable]): A function which takes a token and a
           probability and displays it.
       repetition_penalty (float, optional): The penalty factor for repeating tokens.
       repetition_context_size (int, optional): The number of tokens to consider for repetition penalty.
    """

    if verbose:
        print("=" * 10)
        files = []
        if image is not None:
            files.extend(image)
        if audio is not None:
            files.extend(audio)
        if kwargs.get("video") is not None:
            files.extend(kwargs.get("video"))

        print(f"Files: {files}", "\n")

        print("Prompt:", prompt)

    text = ""
    last_response = None

    eos_tokens = kwargs.get("eos_tokens", None)
    stopping_criteria = kwargs.get("stopping_criteria", None)

    # Get the tokenizer
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Add custom EOS tokens to the stopping criteria
    if eos_tokens is not None:
        tokenizer.stopping_criteria.add_eos_token_ids(eos_tokens)

    # Use custom stopping criteria
    elif stopping_criteria is not None:
        if isinstance(stopping_criteria, StoppingCriteria) or callable(
            stopping_criteria
        ):
            tokenizer.stopping_criteria = stopping_criteria
        else:
            raise ValueError(
                "stopping_criteria must be an instance of StoppingCriteria or a callable"
            )
    else:
        tokenizer.stopping_criteria.reset(model.config.eos_token_id)

    for response in stream_generate(model, processor, prompt, image, audio, **kwargs):
        if verbose:
            print(response.text, end="", flush=True)
        text += response.text
        last_response = response

    if verbose:
        print("\n" + "=" * 10)
        if len(text) == 0:
            print("No text generated for this prompt")
            return GenerationResult(
                text=text,
                token=None,
                logprobs=None,
                prompt_tokens=0,
                generation_tokens=0,
                total_tokens=0,
                prompt_tps=0.0,
                generation_tps=0.0,
                peak_memory=mx.get_peak_memory() / 1e9,
            )
        print(
            f"Prompt: {last_response.prompt_tokens} tokens, "
            f"{last_response.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"Generation: {last_response.generation_tokens} tokens, "
            f"{last_response.generation_tps:.3f} tokens-per-sec"
        )
        print(f"Peak memory: {last_response.peak_memory:.3f} GB")

    return GenerationResult(
        text=text,
        token=last_response.token,
        logprobs=last_response.logprobs,
        prompt_tokens=last_response.prompt_tokens,
        generation_tokens=last_response.generation_tokens,
        total_tokens=last_response.total_tokens,
        prompt_tps=last_response.prompt_tps,
        generation_tps=last_response.generation_tps,
        peak_memory=last_response.peak_memory,
        spec_accepted=last_response.spec_accepted,
        spec_drafted=last_response.spec_drafted,
        spec_cycles=last_response.spec_cycles,
    )


@dataclass
class BatchGenerationResult:
    """
    Result of batch generation with optional image size tracking.

    Attributes:
        texts: Generated text for each sample
        tokens: Last generated token for each sample
        logprobs: Log probabilities for each sample
        prompt_tokens: Number of prompt tokens per sample
        generation_tokens: Number of generated tokens per sample
        total_tokens: Total tokens (prompt + generation) per sample
        prompt_tps: Prompt tokens per second per sample
        generation_tps: Generation tokens per second per sample
        peak_memory: Peak memory usage in GB
        image_sizes: Original (height, width) for each image (for tracking)
    """

    texts: List[str]
    tokens: List[Optional[int]]
    logprobs: List[Optional[List[float]]]
    prompt_tokens: List[int]
    generation_tokens: List[int]
    total_tokens: List[int]
    prompt_tps: List[float]
    generation_tps: List[float]
    peak_memory: float = 0.0
    image_sizes: Optional[List[Tuple[int, int]]] = None


def _left_pad_prompts(prompts, max_length=None):
    if max_length is None:
        max_length = max(len(p) for p in prompts)

    return mx.array([[0] * (max_length - len(p)) + p for p in prompts])


def _make_cache(model, left_padding):
    """
    Convert a list of regular caches into their corresponding
    batch-aware caches.
    """

    def to_batch_cache(c):
        if isinstance(c, cache.KVCache):
            return cache.BatchKVCache(left_padding)
        elif isinstance(c, cache.ArraysCache):
            c.left_padding = mx.array(left_padding)
            return c
        elif isinstance(c, cache.RotatingKVCache):
            if c.keep > 0:
                raise ValueError("RotatingKVCache with keep tokens is not supported.")
            return cache.BatchRotatingKVCache(c.max_size, left_padding)
        elif isinstance(c, cache.CacheList):
            return cache.BatchCacheList(*(to_batch_cache(sub_c) for sub_c in c.caches))
        else:
            raise ValueError(f"{type(c)} does not yet support batching")

    if hasattr(model, "make_cache"):
        model_cache = model.make_cache()
        return [to_batch_cache(c) for c in model_cache]
    else:
        return [cache.BatchKVCache(left_padding) for _ in model.layers]


@dataclass
class BatchStats:
    """
    An data object to hold generation stats.

    Args:
        prompt_tokens (int): The number of prompt tokens processed.
        prompt_tps (float): The prompt processing tokens-per-second.
        prompt_time (float): The time in seconds spent in prompt processing.
        generation_tokens (int): The number of generated tokens.
        generation_tps (float): The tokens-per-second for generation.
        generation_time (float): The time in seconds spent in generation .
        peak_memory (float): The peak memory used so far in GB.
    """

    prompt_tokens: int = 0
    prompt_tps: float = 0
    prompt_time: float = 0
    generation_tokens: int = 0
    generation_tps: float = 0
    generation_time: float = 0
    peak_memory: float = 0


@dataclass
class BatchResponse:
    """
    An data object to hold a batch generation response.

    Args:
        texts: (List[str]): The generated text for each prompt.
        stats (BatchStats): Statistics about the generation.
        image_sizes: (Optional[List[Tuple[int, int]]]): Original (height, width)
            for each image. Useful for tracking which images produced which responses
            and for debugging padding/batching behavior.
    """

    texts: List[str]
    stats: BatchStats
    image_sizes: Optional[List[Tuple[int, int]]] = None


@dataclass
class Batch:
    uids: List[int]
    y: mx.array
    logprobs: mx.array
    max_tokens: List[int]
    num_tokens: List[int]
    cache: List[Any]

    def __len__(self):
        return len(self.uids)

    def filter(self, keep_idx: List[int]):
        self.uids = [self.uids[k] for k in keep_idx]
        self.max_tokens = [self.max_tokens[k] for k in keep_idx]
        self.num_tokens = [self.num_tokens[k] for k in keep_idx]
        keep_idx = mx.array(keep_idx, mx.int32)
        self.y = self.y[keep_idx]
        self.logprobs = self.logprobs[keep_idx]
        for c in self.cache:
            c.filter(keep_idx)

    def extend(self, other):
        self.uids.extend(other.uids)
        self.y = mx.concatenate([self.y, other.y])
        self.logprobs = mx.concatenate([self.logprobs, other.logprobs])
        self.num_tokens.extend(other.num_tokens)
        self.max_tokens.extend(other.max_tokens)
        for c, o in zip(self.cache, other.cache):
            c.extend(o)


class BatchGenerator:

    @dataclass
    class Response:
        uid: int
        token: int
        logprobs: mx.array
        finish_reason: Optional[str]

    def __init__(
        self,
        model,
        processor,
        max_tokens: int = 128,
        stop_tokens: Optional[set] = None,
        sampler: Optional[Callable[[mx.array], mx.array]] = None,
        completion_batch_size: int = 32,
        prefill_batch_size: int = 8,
        prefill_step_size: Optional[int] = 2048,
        prompt_cache=None,
    ):
        self.model = model
        self.unprocessed_prompts = []
        self.max_tokens = max_tokens
        self.processor = processor
        self.tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        self.sampler = sampler or (lambda x: mx.argmax(x, axis=-1))
        self.uid_count = 0
        self.prefill_step_size = prefill_step_size
        self.prefill_batch_size = prefill_batch_size
        self.completion_batch_size = completion_batch_size
        self.prompt_cache = prompt_cache
        self._stats = BatchStats()

        self.tokenizer.stopping_criteria.add_eos_token_ids(stop_tokens)

        self.active_batch = None

    def insert(self, prompts, max_tokens: Union[List[int], int, None] = None):
        uids = []

        if max_tokens is None or isinstance(max_tokens, int):
            max_tokens = [max_tokens or self.max_tokens] * len(prompts)

        for p, m in zip(prompts, max_tokens):
            self.unprocessed_prompts.append((self.uid_count, p, m))
            uids.append(self.uid_count)
            self.uid_count += 1
        # Sort in ascending order of length
        self.unprocessed_prompts = sorted(
            self.unprocessed_prompts, key=lambda x: len(x[1])
        )
        return uids

    def _process_prompts(self, prompts, **kwargs) -> Batch:
        uids, inputs, max_tokens = zip(*prompts)
        lengths = [len(p) for p in inputs]
        max_length = max(lengths)

        self._stats.prompt_tokens += sum(lengths)
        left_padding = [max_length - l for l in lengths]
        inputs = _left_pad_prompts(inputs, max_length=max_length)

        prompt_cache = (
            _make_cache(self.model, left_padding)
            if self.prompt_cache is None
            else self.prompt_cache
        )

        # Slice batch data in kwargs to match current batch size
        batch_size = len(uids)
        for key, value in kwargs.items():
            if isinstance(value, mx.array) and value.ndim > 0:
                kwargs[key] = value[:batch_size]

        inputs_embeds = kwargs.pop("inputs_embeds", None)
        if inputs_embeds is None:
            raise ValueError("inputs_embeds is required")

        if (
            self.prefill_step_size is not None
            and inputs_embeds.shape[1] > self.prefill_step_size
        ):
            # Chunked prefill with embeddings
            while inputs_embeds.shape[1] > 1:
                n_to_process = min(self.prefill_step_size, inputs_embeds.shape[1] - 1)
                self.model(
                    inputs[:, :n_to_process],
                    cache=prompt_cache,
                    inputs_embeds=inputs_embeds[:, :n_to_process],
                    n_to_process=n_to_process,
                    **kwargs,
                )
                mx.eval([c.state for c in prompt_cache])
                inputs_embeds = inputs_embeds[:, n_to_process:]
                inputs = inputs[:, n_to_process:]
                mx.clear_cache()

        y, logprobs = self._step(
            inputs, prompt_cache, inputs_embeds=inputs_embeds, **kwargs
        )

        mx.async_eval(y, logprobs)
        mx.clear_cache()
        return Batch(
            list(uids), y, logprobs, list(max_tokens), [0] * len(uids), prompt_cache
        )

    def _step(self, input_tokens: mx.array, prompt_cache: List[Any], **kwargs):
        output = self.model(input_tokens, cache=prompt_cache, **kwargs)
        logits = output.logits[:, -1, :]
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        sampled = self.sampler(logprobs)

        # TODO: Add KV cache quantization if specified
        return sampled, logprobs

    def stats(self):
        self._stats.prompt_tps = self._stats.prompt_tokens / self._stats.prompt_time
        self._stats.generation_tps = (
            self._stats.generation_tokens / self._stats.generation_time
        )
        self._stats.peak_memory = mx.get_peak_memory() / 1e9
        return self._stats

    def _next(self, **kwargs):
        tic = time.perf_counter()

        prompt_processing = False
        batch = self.active_batch
        num_active = len(batch) if batch else 0
        num_to_add = self.completion_batch_size - num_active
        while num_to_add >= self.prefill_batch_size:
            prompts = self.unprocessed_prompts[: self.prefill_batch_size]
            # Finish processing the last examples of the last batch
            if len(prompts) == 0 and num_active > 0:
                break
            # No more prompts and no more completions, all done
            elif len(prompts) == 0:
                self.active_batch = None
                return []
            # Process prompts
            if batch is not None and not prompt_processing:
                # Finish any active completion tokens
                mx.eval(batch.y, batch.logprobs)
                self._stats.generation_time += time.perf_counter() - tic
                tic = time.perf_counter()

            batch = self._process_prompts(prompts, **kwargs)
            self.unprocessed_prompts = self.unprocessed_prompts[
                self.prefill_batch_size :
            ]
            prompt_processing = True
            # If there was no active batch, set it
            if self.active_batch is None:
                self.active_batch = batch
            else:
                self.active_batch.extend(batch)

            num_active = len(self.active_batch)
            num_to_add -= len(batch)

        batch = self.active_batch
        y, logprobs = batch.y, batch.logprobs
        batch.y, batch.logprobs = self._step(y[:, None], batch.cache)
        mx.async_eval(batch.y, batch.logprobs)

        y = y.tolist()
        toc = time.perf_counter()
        if prompt_processing:
            self._stats.prompt_time += toc - tic
        else:
            self._stats.generation_time += toc - tic
        keep_idx = []
        end_idx = []
        responses = []

        for e, (t, uid, num_tok, max_tok) in enumerate(
            zip(y, batch.uids, batch.num_tokens, batch.max_tokens)
        ):
            num_tok += 1
            batch.num_tokens[e] = num_tok
            if self.tokenizer.stopping_criteria(t):
                finish_reason = "stop"
                end_idx.append(e)
            elif num_tok >= max_tok:
                finish_reason = "length"
                end_idx.append(e)
            else:
                finish_reason = None
                keep_idx.append(e)
            responses.append(self.Response(uid, t, logprobs[e], finish_reason))

        # Remove any finished completions
        if len(end_idx):
            if len(keep_idx) > 0:
                batch.filter(keep_idx)
            else:
                self.active_batch = None

        self._stats.generation_tokens += len(responses)

        if len(responses) > 0 and self._stats.generation_tokens % 100 == 0:
            mx.clear_cache()

        return responses

    def next(self, **kwargs):
        with mx.stream(generation_stream):
            return self._next(**kwargs)


def batch_generate(
    model,
    processor,
    images: Union[str, List[str]] = None,
    audios: Union[str, List[str]] = None,
    prompts: List[str] = None,
    max_tokens: Union[int, List[int]] = 128,
    verbose: bool = False,
    group_by_shape: bool = True,
    track_image_sizes: bool = True,
    **kwargs,
):
    """
    Generate responses for the given batch of prompts with variable-sized images.

    This function implements the transformers-style approach to batching:
    1. Group images with the same shape for efficient batch processing
    2. Process each group as a batch (no padding waste within groups)
    3. Track original image sizes for proper attention masking
    4. Restore results to original batch order

    Key insight: Instead of padding all images to the same spatial dimensions
    (which wastes computation and may hurt accuracy), we group same-sized
    images together so there's zero padding within each group.

    Args:
       model (nn.Module): The language model.
       processor (PreTrainedTokenizer): The tokenizer/processor.
       images (Union[str, List[str]]): Images (paths, URLs, or PIL images).
       audios (Union[str, List[str]]): Audio files (not yet supported for batching).
       prompts (List[str]): The input prompts.
       max_tokens (Union[int, List[int]]): Maximum number of output tokens. This
          can be per prompt if a list is provided.
       verbose (bool): If ``True``, print tokens and timing information.
          Default: ``False``.
       group_by_shape (bool): If ``True``, group same-shaped images for efficient
          batch processing. Default: ``True``.
       track_image_sizes (bool): If ``True``, track and return original image sizes.
          Default: ``True``.
       kwargs: The remaining options get passed to :obj:`BatchGenerator`.
          See :obj:`BatchGenerator` for more details.

    Returns:
        BatchResponse with generated texts, statistics, and optionally image_sizes.
    """
    from PIL import Image

    from .utils import process_image

    processor.detokenizer.reset()
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor

    # Handle single image case
    if isinstance(images, str):
        images = [images]

    # Handle no images case
    if images is None:
        texts, stats = _generate_batch(
            model, processor, prompts, None, max_tokens, verbose, **kwargs
        )
        return BatchResponse(texts, stats)

    # Load and preprocess images
    image_processor = (
        processor.image_processor if hasattr(processor, "image_processor") else None
    )

    processed_images = []
    image_sizes_original = []
    for img in images:
        if isinstance(img, str):
            pil_img = process_image(img, None, image_processor)
        elif isinstance(img, Image.Image):
            pil_img = img
        else:
            pil_img = img
        processed_images.append(pil_img)
        # Track original size
        if hasattr(pil_img, "height"):
            image_sizes_original.append((pil_img.height, pil_img.width))
        else:
            image_sizes_original.append((0, 0))

    # Group images by shape for efficient processing (no padding within groups)
    if group_by_shape and len(processed_images) > 1:
        grouped_images, grouped_indices = group_images_by_shape(processed_images)

        if verbose:
            print(f"[batch_generate] Found {len(grouped_images)} unique image shapes")
    else:
        # Single image or grouping disabled - treat as one group
        shape = (
            (processed_images[0].height, processed_images[0].width)
            if processed_images
            else (0, 0)
        )
        grouped_images = {shape: processed_images}
        grouped_indices = {shape: list(range(len(processed_images)))}

    # Process each shape group
    all_texts = [None] * len(prompts)
    all_image_sizes = [None] * len(prompts)
    total_stats = BatchStats()

    for shape, indices in grouped_indices.items():
        # Get images and prompts for this shape group
        group_images = [processed_images[i] for i in indices]
        group_prompts = [prompts[i] for i in indices]
        group_sizes = [image_sizes_original[i] for i in indices]

        # Handle per-sample max_tokens
        if isinstance(max_tokens, list):
            group_max_tokens = [max_tokens[i] for i in indices]
        else:
            group_max_tokens = max_tokens

        # Process the entire group at once (same shape = no padding needed)
        chunk_texts, chunk_stats = _generate_batch(
            model,
            processor,
            group_prompts,
            group_images,
            group_max_tokens,
            **kwargs,
        )

        # Store results in original order
        for j, orig_idx in enumerate(indices):
            all_texts[orig_idx] = chunk_texts[j]
            all_image_sizes[orig_idx] = group_sizes[j]

        # Accumulate stats
        total_stats.prompt_tokens += chunk_stats.prompt_tokens
        total_stats.prompt_time += chunk_stats.prompt_time
        total_stats.generation_tokens += chunk_stats.generation_tokens
        total_stats.generation_time += chunk_stats.generation_time

    mx.clear_cache()

    # Compute final stats
    if total_stats.prompt_time > 0:
        total_stats.prompt_tps = total_stats.prompt_tokens / total_stats.prompt_time
    if total_stats.generation_time > 0:
        total_stats.generation_tps = (
            total_stats.generation_tokens / total_stats.generation_time
        )
    total_stats.peak_memory = mx.get_peak_memory() / 1e9

    if verbose:
        print(f"[batch_generate] Finished processing {len(prompts)} samples")
        print(
            f"[batch_generate] Prompt: {total_stats.prompt_tokens} tokens, {total_stats.prompt_tps:.3f} tokens-per-sec"
        )
        print(
            f"[batch_generate] Generation: {total_stats.generation_tokens} tokens, "
            f"{total_stats.generation_tps:.3f} tokens-per-sec"
        )
        print(f"[batch_generate] Peak memory: {total_stats.peak_memory:.3f} GB")

    response = BatchResponse(all_texts, total_stats)
    if track_image_sizes:
        response.image_sizes = all_image_sizes
    return response


def _generate_batch(
    model,
    processor,
    prompts: List[str],
    images: List = None,
    max_tokens: Union[int, List[int]] = 100,
    verbose: bool = False,
    **kwargs,
) -> Tuple[List[str], BatchStats]:

    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    batch_size = len(prompts)

    num_images_list = [
        1 if i < (len(images) if images is not None else 0) else 0
        for i in range(len(prompts))
    ]
    formatted_prompts = [
        apply_chat_template(
            processor,
            model.config,
            p,
            num_images=num_images_list[i],
        )
        for i, p in enumerate(prompts)
    ]

    add_special_tokens = (
        not hasattr(processor, "chat_template")
        if model.config.model_type in ["gemma3", "gemma3n"]
        else True
    )

    resize_shape = kwargs.pop("resize_shape", None)
    image_token_index = getattr(model.config, "image_token_index", None)

    inputs = prepare_inputs(
        processor,
        images=images,
        audio=None,
        prompts=formatted_prompts,
        image_token_index=image_token_index,
        resize_shape=resize_shape,
        add_special_tokens=add_special_tokens,
        pad_to_uniform_size=False,  # Since images are pre-grouped by shape, they're already uniform size
    )
    input_ids = inputs.get("input_ids", None)
    pixel_values = inputs.get("pixel_values", None)
    mask = inputs.get("attention_mask", None)

    data_kwargs = {
        k: v
        for k, v in inputs.items()
        if k not in ["input_ids", "pixel_values", "attention_mask"]
    }

    # Use batch_size for prefill and completion to ensure consistent processing
    gen = BatchGenerator(
        model.language_model,
        processor,
        prefill_batch_size=batch_size,
        completion_batch_size=batch_size,
        **kwargs,
    )

    with wired_limit(model, [generation_stream]):

        embedding_output = model.get_input_embeddings(
            input_ids, pixel_values, mask=mask, **data_kwargs
        )

        gen_kwargs = {**data_kwargs, **embedding_output.to_dict()}

        uids = gen.insert(input_ids.tolist(), max_tokens)
        results = {uid: [] for uid in uids}
        while responses := gen.next(**gen_kwargs):
            for r in responses:
                if r.finish_reason != "stop":
                    results[r.uid].append(r.token)

    texts = [tokenizer.decode(results[uid]) for uid in uids]
    return texts, gen.stats()


def main():
    args = parse_arguments()
    if isinstance(args.image, str):
        args.image = [args.image]

    model, processor = load(
        args.model,
        args.adapter_path,
        revision=args.revision,
        trust_remote_code=args.trust_remote_code,
        quantize_activations=args.quantize_activations,
    )
    config = model.config

    prompt = args.prompt

    num_images = len(args.image) if args.image is not None else 0
    num_audios = (
        1 if args.audio is not None else 0
    )  # TODO: Support multiple audio files
    prompt = apply_chat_template(
        processor, config, prompt, num_images=num_images, num_audios=num_audios
    )

    if args.early_exit_layer is not None and args.draft_model is not None:
        raise ValueError("Cannot use both --early-exit-layer and --draft-model")

    if args.early_exit_layer is not None:
        n_layers = len(model.language_model.layers)
        if not (1 <= args.early_exit_layer < n_layers):
            raise ValueError(
                f"--early-exit-layer must be between 1 and {n_layers - 1} "
                f"(model has {n_layers} layers), got {args.early_exit_layer}"
            )

    # Load draft model if specified
    draft_model_obj = None
    if args.draft_model is not None:
        draft_model_obj, draft_processor = load(
            args.draft_model,
            revision=args.revision,
            trust_remote_code=args.trust_remote_code,
            quantize_activations=args.quantize_activations,
        )
        # Validate tokenizer vocab compatibility
        main_tokenizer = (
            processor.tokenizer if hasattr(processor, "tokenizer") else processor
        )
        draft_tokenizer = (
            draft_processor.tokenizer
            if hasattr(draft_processor, "tokenizer")
            else draft_processor
        )
        if len(main_tokenizer) != len(draft_tokenizer):
            print(
                f"[WARNING] Tokenizer vocab size mismatch: "
                f"main={len(main_tokenizer)}, draft={len(draft_tokenizer)}. "
                f"Speculative decoding may produce incorrect results."
            )

    kwargs = {}

    if args.early_exit_layer is not None:
        kwargs["early_exit_layer"] = args.early_exit_layer

    if draft_model_obj is not None:
        kwargs["draft_model"] = draft_model_obj

    if args.num_draft_tokens != 3:
        kwargs["num_draft_tokens"] = args.num_draft_tokens

    if args.resize_shape is not None:
        if len(args.resize_shape) not in [1, 2]:
            raise ValueError("Resize shape must be 1 or 2 integers")
        kwargs["resize_shape"] = (
            (args.resize_shape[0],) * 2
            if len(args.resize_shape) == 1
            else tuple(args.resize_shape)
        )

    if args.eos_tokens is not None:
        eos_tokens = []
        for token in args.eos_tokens:
            try:
                decoded_token = codecs.decode(token, "unicode_escape")
                eos_tokens.append(decoded_token)
            except (UnicodeDecodeError, UnicodeError):
                eos_tokens.append(token)
        kwargs["eos_tokens"] = eos_tokens

    if args.skip_special_tokens:
        kwargs["skip_special_tokens"] = args.skip_special_tokens

    # Add processor kwargs from JSON
    if args.processor_kwargs:
        kwargs.update(args.processor_kwargs)

    if args.chat:
        chat = []
        if args.system:
            chat.append({"role": "system", "content": args.system})
        while user := input("User:"):
            chat.append({"role": "user", "content": user})
            prompt = apply_chat_template(processor, config, chat, num_images=num_images)
            response = ""
            print("Assistant:", end="")
            stream_kwargs = {
                "max_tokens": args.max_tokens,
                "temperature": args.temperature,
                **kwargs,
            }
            if args.prefill_step_size is not None:
                stream_kwargs["prefill_step_size"] = args.prefill_step_size

            for chunk in stream_generate(
                model,
                processor,
                prompt,
                args.image,
                args.audio,
                **stream_kwargs,
            ):
                response += chunk.text
                print(chunk.text, end="")

            chat.append({"role": "assistant", "content": response})
            print()

    else:
        gen_kwargs = {
            "image": args.image,
            "audio": args.audio,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "verbose": args.verbose,
            "max_kv_size": args.max_kv_size,
            "kv_bits": args.kv_bits,
            "kv_group_size": args.kv_group_size,
            "quantized_kv_start": args.quantized_kv_start,
            **kwargs,
        }
        if args.prefill_step_size is not None:
            gen_kwargs["prefill_step_size"] = args.prefill_step_size

        result = generate(
            model,
            processor,
            prompt,
            **gen_kwargs,
        )
        if not args.verbose:
            print(result.text)


if __name__ == "__main__":
    print(
        "Calling `python -m mlx_vlm.generate ...` directly is deprecated."
        " Use `mlx_vlm generate` or `python -m mlx_vlm generate` instead."
    )
    main()
