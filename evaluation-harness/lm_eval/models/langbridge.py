import math
import torch
import torch.nn.functional as F
import transformers
import peft
from peft import __version__ as PEFT_VERSION
from pathlib import Path
from typing import List, Mapping, NewType, Optional, Tuple, Union
from tqdm import tqdm

from transformers import BatchEncoding

from lm_eval import utils
from lm_eval.base import BaseLM
from langbridge import LangBridgeModel

TokenSequence = Union[List[int], torch.LongTensor, torch.Tensor, BatchEncoding]

_DeviceMapping = NewType(
    "DeviceMapping", Mapping[str, Union[int, str, torch.device]])


def _get_accelerate_args(
    low_cpu_mem_usage: Optional[bool] = True,
    device_map_option: Optional[str] = "auto",
    max_memory_per_gpu: Optional[Union[int, str]] = None,
    max_cpu_memory: Optional[Union[int, str]] = None,
    offload_folder: Optional[str] = "./offload",
) -> dict:
    """Returns the kwargs needed to apply `accelerate` in `AutoModel.from_pretrained`."""
    max_memory = {}
    if max_memory_per_gpu is not None:
        max_memory_per_gpu_map = {
            device_idx: max_memory_per_gpu
            for device_idx in range(torch.cuda.device_count())
        }
        max_memory.update(max_memory_per_gpu_map)
    if max_cpu_memory is not None:
        max_memory["cpu"] = max_cpu_memory

    args = {}
    if max_memory:
        args["max_memory"] = max_memory
    args["low_cpu_mem_usage"] = low_cpu_mem_usage
    args["device_map"] = device_map_option
    args["offload_folder"] = offload_folder
    return args


def _get_dtype(
    dtype: Union[str, torch.dtype], config: Optional[transformers.AutoConfig] = None
) -> torch.dtype:
    """Converts `dtype` from `str` to torch.dtype when possible."""
    if dtype is None and config is not None:
        _torch_dtype = config.torch_dtype
    elif isinstance(dtype, str) and dtype != "auto":
        # Convert `str` args torch dtype: `float16` -> `torch.float16`
        _torch_dtype = getattr(torch, dtype)
    else:
        _torch_dtype = dtype
    return _torch_dtype


class HuggingFaceLB(BaseLM):
    AUTO_CONFIG_CLASS: transformers.AutoConfig = transformers.AutoConfig
    AUTO_TOKENIZER_CLASS: transformers.AutoTokenizer = transformers.AutoTokenizer
    AUTO_MODEL_CLASS: transformers.AutoModel = None
    AUTO_PEFT_CLASS: peft.PeftModel = None

    # Default max sequence length setting for when no `max_length` is provided
    # or no max length config setting is found in the model or tokenizer.
    _DEFAULT_MAX_LENGTH: int = 2048

    def __init__(
        self,
        model: LangBridgeModel,
        enc_tokenizer: Optional[str] = None,
        lm_tokenizer: Optional[str] = None,
        subfolder: Optional[str] = None,
        revision: Optional[str] = "main",
        batch_size: Optional[Union[int, str]] = 1,
        max_batch_size: Optional[int] = 512,
        max_gen_toks: Optional[int] = 1024,
        max_length: Optional[int] = None,
        add_special_tokens: Optional[bool] = None,
        use_accelerate: Optional[bool] = False,
        device: Optional[Union[int, str]] = "cuda",
        load_in_8bit: Optional[bool] = False,
        load_in_4bit: Optional[bool] = False,
        trust_remote_code: Optional[bool] = False,
    ):
        """Initializes a HuggingFace `AutoModel` and `AutoTokenizer` for evaluation.
        Args:
            pretrained (str):
                The HuggingFace Hub model ID name or the path to a pre-trained
                model to load. This is effectively the `pretrained_model_name_or_path`
                argument of `from_pretrained` in the HuggingFace `transformers` API.
            quantized (str or bool, optional, defaults to False):
                File name of a GPTQ quantized model to load. Set to `True` to use the
                default name of the quantized model.
            add_special_tokens (bool, optional, defaults to True):
                Whether to add special tokens to the input sequences. If `None`, the
                default value will be set to `True` for seq2seq models (e.g. T5) and
                `False` for causal models.
                WARNING: Evaluating causal models with `add_special_tokens=True` is
                currently __not__ supported.
            > Large model loading `accelerate` arguments
            use_accelerate (bool, optional, defaults to False):
                If True, uses the `accelerate` library to load a large model across
                multiple devices.
            low_cpu_mem_usage (bool, optional, defaults to True):
                It True, uses the `accelerate` library to accelerate loading the model.
            device_map_option (str, optional, defaults to "auto"):
                The device map option to use when loading the model with
                `accelerate`.
                Options:
                    "auto", "balanced", "balanced_low_0", "sequential"
                See the `accelerate` docs for more details on these options:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.device_map
            max_memory_per_gpu (Union[int, str], optional, defaults to None):
                The maximum memory available for each GPU in bytes as `int` or in
                the format f"{significand}{unit_symbol}" where {unit_symbol} is
                any of ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in
                the "Parameters for big model inference" section of the following
                docs:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.max_memory
            max_cpu_memory (Union[int, str], optional, defaults to None):
                The maximum available CPU RAM in bytes as `int` or in the format
                f"{significand}{unit_symbol}" where {unit_symbol} is any of
                ["GB", "MB", "GIB", "MIB"]. Refer to the `max_memory` arg in the
                "Parameters for big model inference" section of the following docs:
                https://huggingface.co/docs/transformers/main/en/main_classes/model#transformers.PreTrainedModel.from_pretrained.max_memory
            offload_folder (str, optional, defaults to "./offload"):
                The folder to offload weights into if `device_map` contains any
                "disk" value.
            dtype (Union[str, torch.dtype], optional, defaults to None):):
                Converts the model weights to `dtype`, if specified. Strings get
                converted to `torch.dtype` objects (e.g. `float16` -> `torch.float16`).
                Use `dtype="auto"` to derive the type from the model’s weights.
            peft (str, optional, defaults to None):
                Path of the adapter weights to load from Huggingface. This will usually
                include a directory that includes the files `adapter_config.json` and
                `adapter_model.bin`. Compatible with [PEFT](https://github.com/huggingface/peft)
            load_in_8bit (bool, optional, defaults to False):
                If True, will convert the loaded model into mixed-8bit quantized model. See:
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#load-a-large-model-in-8bit
            load_in_4bit (bool, optional, defaults to False):
                If True, will convert the loaded model into mixed-4bit quantized model. See:
                https://huggingface.co/docs/transformers/main/en/main_classes/quantization#load-a-large-model-in-4bit
            trust_remote_code (bool, optional, defaults to False):
                If True, will trust the remote code when loading the model.
            gptq_use_triton (bool, optional, defaults to False):
                Use Triton for GPTQ inference.
            inject_fused_attention (bool, optional, defaults to True):
                Inject fused attention into GPTQ model.
            bnb_4bit_quant_type (str, optional, defaults to None):
                The quantization type to use for BnB 4bit quantization. See:
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L77
            bnb_4bit_compute_dtype (Union[str, torch.dtype], optional, defaults to None):
                The compute dtype to use for BnB 4bit quantization. See:
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L74
            bnb_4bit_use_double_quant (bool, optional, defaults to False):
                Whether or not to use double quant to quantize the absmax.
                https://github.com/huggingface/transformers/blob/main/src/transformers/utils/quantization_config.py#L80

        """
        super().__init__()
        assert isinstance(device, str)
        assert isinstance(batch_size, (int, str))
        if (
            add_special_tokens is not None
            and self.AUTO_MODEL_CLASS is transformers.AutoModelForCausalLM
        ):
            # TODO: Support evaluating causal models with special tokens. Currently,
            # this is not possible because the `_loglikelihood_tokens()` method for
            # causal LMs makes a no-special-tokens assumption given that contexts
            # and labels/continuations are tokenized separately without special
            # tokens, concatenated, and then processed as inputs.
            assert (
                not add_special_tokens
            ), "Evaluating causal models with `add_special_tokens=True` is currently not supported."

        # setup for automatic batch size detection
        if str(batch_size).startswith("auto"):
            batch_size = batch_size.split(":")
            self._batch_size = batch_size[0]
            self.batch_schedule = float(
                batch_size[1]) if len(batch_size) > 1 else 1
        else:
            self._batch_size = int(batch_size)
        self.max_batch_size = max_batch_size

        self._max_gen_toks = max_gen_toks
        self._max_length = max_length
        self._config = model.config

        self._add_special_tokens = add_special_tokens
        self.enc_tokenizer = enc_tokenizer
        self.lm_tokenizer = lm_tokenizer
        self.enc_tokenizer.model_max_length = self.max_length
        self.lm_tokenizer.model_max_length = self.max_length

        self.model = model
        self.model.eval()
        torch.set_grad_enabled(False)

        self._device = device
        if use_accelerate and "lm_head" in self.model.hf_device_map:
            # `accelerate` can place `lm_head` weights on a different device than
            # the user specified one so we force `self._device` to be the same as
            # `lm_head`'s.
            self._device = self.model.hf_device_map["lm_head"]
        if not use_accelerate and not (load_in_4bit or load_in_8bit):
            try:
                self.model.to(self._device)
            except:
                print("Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes`. If the desired GPU is being used, this message is safe to ignore.")

    @property
    def add_special_tokens(self) -> bool:
        return True

    @property
    def eot_token(self) -> str:
        return self.lm_tokenizer.eos_token

    @property
    def eot_token_id(self) -> int:
        return self.lm_tokenizer.eos_token_id

    @property
    def max_gen_toks(self) -> int:
        return self._max_gen_toks

    @property
    def max_length(self) -> int:
        """Return the maximum sequence length of the model.
        NOTE: Different model configurations have different max sequence length
        attribute names.
            - n_positions: (CTRLConfig, T5Config)
            - max_position_embeddings: (BartConfig, RoFormerConfig)
            - n_ctx: (GPT2Config)
        NOTE: For relative position encoded models you should specify the max
        sequence length of the model in the constructor via `max_length`.
        """
        if self._max_length is not None:
            return self._max_length
        # Try to get the sequence length from the model config.
        seqlen_config_attrs = (
            "n_positions", "max_position_embeddings", "n_ctx")
        for attr in seqlen_config_attrs:
            if hasattr(self._config, attr):
                return getattr(self._config, attr)
        if hasattr(self.lm_tokenizer, "model_max_length"):
            if self.lm_tokenizer.model_max_length == 1000000000000000019884624838656:
                return self._DEFAULT_MAX_LENGTH
            return self.lm_tokenizer.model_max_length
        return self._DEFAULT_MAX_LENGTH

    @property
    def batch_size(self) -> int:
        # TODO: Add adaptive batch size.
        return self._batch_size  # * gpus

    @property
    def device(self) -> Union[int, str, torch.device]:
        return self._device

    def enc_tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.enc_tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def enc_tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.enc_tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_encode(self, string: str) -> TokenSequence:
        # TODO: Merge `tok_encode_batch` here.
        return self.lm_tokenizer.encode(string, add_special_tokens=self.add_special_tokens)

    def tok_encode_batch(self, strings: List[str]) -> TokenSequence:
        return self.lm_tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_encode_batch_enc(self, strings: List[str]) -> TokenSequence:
        return self.enc_tokenizer(
            strings,
            padding=True,
            add_special_tokens=self.add_special_tokens,
            return_tensors="pt",
        )

    def tok_decode(self, tokens: torch.LongTensor) -> List[str]:
        return self.lm_tokenizer.batch_decode(tokens, skip_special_tokens=True)

    def greedy_until(
        self, requests: List[Tuple[str, Union[List[str], str]]]
    ) -> List[str]:
        def _collate(x):
            tokens = self.tok_encode(x[0])
            return len(tokens), x[0]

        results = []
        reorder = utils.Reorderer(requests, _collate)

        adaptive_batch_size = None
        if self.batch_size == "auto":
            # using rolling window with maximum context
            print("Passed argument batch_size = auto. Detecting largest batch size")
            batch_size = self._detect_batch_size()
            print(f"Determined Largest batch size: {batch_size}")
            adaptive_batch_size = batch_size

        for chunk in utils.chunks(
            tqdm(reorder.get_reordered(), disable=False),
            self.batch_size if self.batch_size != "auto" else adaptive_batch_size,
        ):
            context = [c[0] for c in chunk]
            request_args = chunk[0][1]
            stop = request_args.get("until", None)
            stop_sequences = stop if isinstance(stop, list) else [stop]
            max_generation_length = request_args.get("max_length", None)

            assert (
                isinstance(max_generation_length,
                           int) or max_generation_length is None
            )
            assert isinstance(stop_sequences, list) or stop_sequences is None

            # TODO: Find a better way to handle stop sequences for 0-shot.
            if stop_sequences is None:
                until = [self.eot_token]
            else:
                until = stop_sequences + [self.eot_token]

            if max_generation_length is None:
                max_tokens = self.max_gen_toks
            else:
                max_tokens = max_generation_length

            token_context = self.tok_encode_batch_enc(context)

            responses = self._model_generate(
                inputs=token_context,
                max_tokens=max_tokens,
                stop=until,
            )
            responses = self.tok_decode(responses.tolist())

            for i, response in enumerate(responses):
                # Ensure the generated responses do not contain the stop sequences.
                for term in until:
                    response = response.split(term)[0]
                # partial caching
                self.cache_hook.add_partial(
                    "greedy_until", (context, until), response)

                print('[Context]: ', context[i])
                print('Response: ', response)
                # print('[Original Response]: ', responses[i])
                results.append(response)
        return reorder.get_original(results)


class LBSeq2SeqLM(HuggingFaceLB):
    """Seq2Seq language modeling.
    You can find a set of supported models in the following documentation:
    https://huggingface.co/docs/transformers/main/model_doc/auto#transformers.AutoModelForSeq2SeqLM
    """

    def loglikelihood(
        self, requests: List[Tuple[str, str]]
    ) -> List[Tuple[float, bool]]:
        new_requests = []
        for chunk in utils.chunks(requests, self.batch_size):
            context, continuation = zip(*chunk)

            # Fill empty contexts with the EOT token.
            context = [
                f"{self.eot_token}" if len(text) == 0 else text for text in context
            ]
            context_enc = self.enc_tok_encode_batch(context)
            for key in context_enc:
                context_enc[key] = context_enc[key][:, -self.max_length:]

            # Remove leading whitespace introduced by the default
            # `text_target_separator` since the context and continuation
            # will not be concatenated as a single (decoder) input.
            continuation = [text.lstrip() for text in continuation]
            continuation_enc = self.tok_encode_batch(list(continuation))
            for key in continuation_enc:
                continuation_enc[key] = continuation_enc[key][:, -
                                                              self.max_length:]

            new_requests.append(
                ((context, continuation), context_enc, continuation_enc)
            )
        return self._loglikelihood_tokens(new_requests)

    def _loglikelihood_tokens(
        self,
        requests: List[Tuple[Tuple[str, str], TokenSequence, TokenSequence]],
        disable_tqdm: Optional[bool] = False,
    ) -> List[Tuple[float, bool]]:
        results = []
        for chunk in tqdm(
            requests, total=math.ceil(len(requests)), disable=disable_tqdm
        ):
            cache_keys, inputs_tokens, targets_tokens = chunk
            inputs_tokens = inputs_tokens.to(self.device)
            targets_tokens = targets_tokens.to(self.device)
            outputs = self._model_call(
                inputs=inputs_tokens, labels=targets_tokens)
            log_softmaxes = F.log_softmax(outputs.logits, dim=-1)

            output_iterator = zip(
                zip(cache_keys[0], cache_keys[1]),
                log_softmaxes,
                targets_tokens["input_ids"],
                targets_tokens["attention_mask"],
            )
            _, enc_input_length = inputs_tokens['input_ids'].shape
            # TODO calculating token probabilities on soft prompts
            for cache_key, log_softmax, target_tokens, target_mask in output_iterator:
                length = target_mask.sum()
                # +1 for the added eos token in LinearWithAddedEos
                start_idx = enc_input_length
                end_idx = start_idx + length

                log_softmax = log_softmax[start_idx:end_idx]
                target_tokens = target_tokens[:length]
                greedy_tokens = log_softmax.argmax(dim=-1)
                max_equal = (greedy_tokens == target_tokens).all()
                target_logits = torch.gather(
                    log_softmax, 1, target_tokens.unsqueeze(-1)
                ).squeeze(-1)
                answer = (float(target_logits.sum()), bool(max_equal))
                results.append(answer)
                if cache_key is not None:
                    self.cache_hook.add_partial(
                        "loglikelihood", cache_key, answer)
        return results

    def _model_call(
        self, inputs: TokenSequence, labels: Optional[TokenSequence] = None
    ) -> TokenSequence:

        return self.model(
            enc_ids=inputs["input_ids"],
            enc_mask=inputs["attention_mask"],
            input_ids=labels["input_ids"],
            attention_mask=labels["attention_mask"],
            labels=labels["input_ids"],
        )

    def _model_generate(
        self,
        inputs: transformers.BatchEncoding,
        max_tokens: int,
        stop: Optional[List[str]] = None,
    ) -> TokenSequence:
        enc_ids = inputs["input_ids"][:, -self.max_length:].to(self.device)
        enc_mask = inputs["attention_mask"][:, -
                                            self.max_length:].to(self.device)

        # Generate one token to calculate the number of start tokens prepended to decoder_input_ids
        # (leaving this here in case the below assumption is violated in the future)
        # one_tok_gen = self.model.generate(
        #    input_ids=torch.zeros((1, 1), dtype=torch.int),
        #    min_length=2,
        #    max_new_tokens=1,
        # ).squeeze()
        # initial_decoder_input_length = len(one_tok_gen) - 1

        # Assume that there will always only be one token in the decoder inputs, assumption holds for existing HF models
        input_ids = torch.LongTensor([self.lm_tokenizer.bos_token_id])
        input_ids = input_ids.repeat(enc_ids.shape[0], 1).to(self.device)
        attention_mask = torch.ones_like(input_ids)

        stopping_criteria = stop_sequences_criteria(
            self.lm_tokenizer, stop, 1, input_ids.shape[0]
        )

        generations = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            enc_ids=enc_ids,
            enc_mask=enc_mask,
            max_new_tokens=max_tokens,
            stopping_criteria=stopping_criteria,
            do_sample=False,
        )
        return generations


class MultiTokenEOSCriteria(transformers.StoppingCriteria):
    """Criteria to stop on the specified multi-token sequence."""

    def __init__(
        self,
        sequence: str,
        tokenizer: transformers.PreTrainedTokenizer,
        initial_decoder_input_length: int,
        batch_size: int,
    ):
        self.initial_decoder_input_length = initial_decoder_input_length
        self.done_tracker = [False] * batch_size
        self.sequence = sequence

        # print(sequence)
        self.sequence_ids = tokenizer.encode(
            sequence, add_special_tokens=False)

        # have no idea why tokenizer adds a space in front
        # a temporary fix is to remove the first token
        if 29871 in self.sequence_ids:
            self.sequence_ids = self.sequence_ids[1:]

        # print(type(tokenizer))
        # print('Stopping Sequence:', self.sequence)
        # print('Stopping Sequence IDs:', self.sequence_ids)
        self.sequence_id_len = len(self.sequence_ids)
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        # For efficiency, we compare the last n tokens where n is the number of tokens in the stop_sequence
        lookback_ids_batch = input_ids[:, self.initial_decoder_input_length:][
            :, -self.sequence_id_len:
        ]

        lookback_tokens_batch = self.tokenizer.batch_decode(lookback_ids_batch)

        for i, done in enumerate(self.done_tracker):
            if not done:
                self.done_tracker[i] = self.sequence in lookback_tokens_batch[i]
        return False not in self.done_tracker


def stop_sequences_criteria(
    tokenizer: transformers.PreTrainedTokenizer,
    stop_sequences: List[str],
    initial_decoder_input_length: int,
    batch_size: int,
) -> transformers.StoppingCriteriaList:
    return transformers.StoppingCriteriaList(
        [
            *[
                MultiTokenEOSCriteria(
                    sequence, tokenizer, initial_decoder_input_length, batch_size
                )
                for sequence in stop_sequences
            ],
        ]
    )
