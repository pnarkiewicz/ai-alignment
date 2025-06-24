from __future__ import annotations

import base64
import copy
import io
import math
import os
import random
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import PeftModel
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, GenerationConfig

import utils.constants as constants
from models.model import (
    GenerationParams,
    Model,
    ModelInput,
    ModelResponse,
    ProbeHyperparams,
    SpeechStructure,
)
from models.openai_model import OpenAIModel
from prompts import RoleType
from utils import get_device, logger_utils, string_utils, timer

HAVE_FLASH = False
try:
    import flash_attn

    HAVE_FLASH = True
except ImportError:
    print("Flash attention not available")


class LLMInput(BaseModel):
    instruction: str
    input: str
    extra_suffix: str | None = None


class LLModel(Model):
    INSTRUCTION_PREFIX = ""
    INSTRUCTION_SUFFIX = ""
    ATTENTION_MODULES = []
    MLP_MODULES = []
    TARGET_MODULES = []
    DEFAULT_GENERATION_PARAMS = GenerationParams()
    MAX_MINI_BATCH_SIZE = 8
    QUANTIZE = True

    def __init__(
        self,
        alias: str,
        file_path: str | None = None,
        is_debater: bool = True,
        nucleus: bool = True,
        instruction_prefix: str = "",
        instruction_suffix: str = "",
        requires_file_path: bool = True,
        probe_hyperparams: ProbeHyperparams | None = None,
        max_mini_batch_size: int | None = None,
        tokenizer_file_path: str | None = None,
        quantize: bool = True,
        generation_params: GenerationParams | None = None,
        peft_base_model: str | None = None,
        **kwargs,
    ):
        """
        An LLModel uses a large language model (currently Llama 2 or Mistral) to generate text.

        Args:
            alias: String that identifies the model for metrics and deduplication
            file_path: the name of the huggingface model to load
            is_debater: Boolean indicating whether the model is a debater (true) or judge (false)
            nucleus: Whether nucleus sampling (true) or beam_search (false) should be used.
            instruction_prefix: the prefix to use before the instructions that get passed to the model
            instruction_suffix: the suffix to use after the instructions that get passed to the model
            requires_file_path: whether a file path is needed to instantiate the model
            probe_hyperparams: configuration for a linear probe judge
            max_mini_batch_size: maximum number of elements before the batch gets split
            tokenizer_file_path: if the tokenizer has a separate file path, fill this in.
                Defaults to the same as the file_path
        """
        if generation_params is None:
            generation_params = GenerationParams()
        super().__init__(alias=alias, is_debater=is_debater)
        torch.cuda.empty_cache()
        self.logger = logger_utils.get_default_logger(__name__)
        self.instruction_prefix = instruction_prefix
        self.instruction_suffix = instruction_suffix
        self.instantiated_model = False
        self.max_mini_batch_size = max_mini_batch_size or LLModel.MAX_MINI_BATCH_SIZE
        self.quantize = quantize
        self.generation_params = generation_params
        if file_path or not requires_file_path:
            self.instantiated_model = True
            self.is_debater = is_debater
            self.tokenizer_file_path = tokenizer_file_path or file_path

            self.tokenizer, self.model = self.instantiate_tokenizer_and_hf_model(
                file_path=file_path,
                tokenizer_file_path=tokenizer_file_path,
                quantize=quantize,
                peft_base_model=peft_base_model,
            )
            self.generation_config = self.create_default_generation_config(
                is_debater=is_debater, generation_params=self.generation_params
            )

            if not nucleus:
                self.generation_config.num_beams = 2
                self.generation_config.do_sample = False
                self.generation_config.top_p = None
                self.generation_config.temperature = None

            if probe_hyperparams:
                if not is_debater:
                    self.model = LLModuleWithLinearProbe(
                        base_model=self.model,
                        linear_idxs=probe_hyperparams.linear_idxs,
                        file_path=probe_hyperparams.file_path,
                    )
                else:
                    self.logger.warn("Probe hyperparameters were passed in for a debater model. This is not supported.")
        else:
            self.is_debater = False
            self.tokenizer = None
            self.model = None
            self.generation_config = None

    def create_default_generation_config(
        self, is_debater: bool, generation_params: GenerationParams
    ) -> GenerationConfig:
        """Creates a default generation config so that the model can generate text"""
        config_terms = {
            "max_new_tokens": generation_params.max_new_tokens,
            "num_return_sequences": 1,
            "output_scores": True,
            "return_dict_in_generate": True,
            "do_sample": generation_params.do_sample,
            "use_cache": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": [self.tokenizer.eos_token_id],
            "output_hidden_states": not is_debater,
            "min_length": -1,
            "top_k": 0.0,
            "top_p": 1.0,
        }
        if is_debater:
            if generation_params.do_sample:
                config_terms["temperature"] = generation_params.temperature
                config_terms["top_p"] = generation_params.top_p

            if generation_params.use_generation_penalties:
                config_terms["repetition_penalty"] = generation_params.repetition_penalty
                config_terms["exponential_decay_length_penalty"] = (
                    generation_params.max_new_tokens * 2 // 3,
                    1.1,
                )

        return GenerationConfig(**config_terms)

    @classmethod
    def instantiate_tokenizer(
        self, file_path: str, requires_token: bool = False
    ) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        tokenizer = AutoTokenizer.from_pretrained(
            file_path,
            token=os.getenv("META_ACCESS_TOKEN") if requires_token else None,
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    @classmethod
    def get_bnb_config(cls) -> BitsAndBytesConfig:
        return None
        # return BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        # )

    @classmethod
    def instantiate_hf_model(
        self,
        file_path: str,
        requires_token: bool = False,
        use_cache: bool = True,
        quantize: bool = True,
        peft_base_model: str | None = None,
    ) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}

        device = get_device()
        quantize = False
        if not torch.cuda.is_available():
            quantize = False
            print("CUDA not available, disabling quantization")
            if torch.backends.mps.is_available():
                device_map = {"": "mps"}  # Prevents trl from failing on MPS

        model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=peft_base_model or file_path,
            device_map="auto",
            trust_remote_code=True,
            use_flash_attention_2=HAVE_FLASH,
            use_cache=use_cache,
            token=os.getenv("META_ACCESS_TOKEN") if requires_token else None,
            quantization_config=LLModel.get_bnb_config() if quantize else None,
            torch_dtype=None if quantize else torch.bfloat16,
        )#.to(device)

        if peft_base_model:
            model = PeftModel.from_pretrained(
                model=model, model_id=file_path, adapter_name="default", is_trainable=False
            )
            model = model.merge_and_unload()

        return model

    def instantiate_tokenizer_and_hf_model(
        self,
        file_path: str,
        requires_token: bool = False,
        tokenizer_file_path: str | None = "",
        quantize: bool = True,
        peft_base_model: str | None = None,
    ) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Constructs the tokenizer and huggingface model at the specified filepath"""
        tokenizer = LLModel.instantiate_tokenizer(
            file_path=tokenizer_file_path or file_path, requires_token=requires_token
        )
        hf_model = LLModel.instantiate_hf_model(
            file_path=file_path,
            requires_token=requires_token,
            quantize=quantize,
            peft_base_model=peft_base_model,
        )
        return tokenizer, hf_model

    @classmethod
    def generate_llm_input_from_model_inputs(cls, input_list: list[ModelInput], extra_suffix: str = "") -> LLMInput:
        """Converts a ModelInput into the LLMInput that's expected by the model"""
        return LLMInput(
            instruction="\n".join(
                model_input.content for model_input in filter(lambda x: x.role == RoleType.SYSTEM, input_list)
            ),
            input="\n".join(
                model_input.content for model_input in filter(lambda x: x.role != RoleType.SYSTEM, input_list)
            ),
            extra_suffix=extra_suffix,
        )

    @classmethod
    def generate_input_str(cls, llm_input: LLMInput, instruction_prefix: str = "", instruction_suffix: str = "") -> str:
        """Transforms a LLMInput into a standardized format"""
        return "{} {}\n\n{} {}{}".format(
            instruction_prefix,
            llm_input.instruction,
            llm_input.input,
            instruction_suffix,
            (" " + llm_input.extra_suffix) if llm_input.extra_suffix else "",
        )

    @classmethod
    def convert_to_input_string(
        cls,
        input_list: list[ModelInput],
        tokenizer: AutoTokenizer,
        speech_structure: SpeechStructure,
    ) -> str:
        """Converts the list of model inputs to a string"""

        system = "\n".join(
            model_input.content for model_input in filter(lambda x: x.role == RoleType.SYSTEM, input_list)
        )
        user = "\n".join(model_input.content for model_input in filter(lambda x: x.role != RoleType.SYSTEM, input_list))
        try:
            output = tokenizer.apply_chat_template(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                tokenize=False,
                add_generation_prompt=True,
                chat_template=tokenizer.chat_template,
            )
            return output
        except Exception:
            return LLModel.generate_input_str(
                llm_input=LLMInput(
                    instruction=system,
                    input=user,
                    extra_suffix="",
                ),
                instruction_prefix=cls.INSTRUCTION_PREFIX,
                instruction_suffix=cls.INSTRUCTION_SUFFIX,
            )

    def generate_input_strs(
        self,
        inputs: list[list[ModelInput]],
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
    ) -> list[str]:
        """Converts a list of model inputs into a list of strings that can be tokenized"""
        return [
            LLModel.convert_to_input_string(
                input_list=input_list, tokenizer=self.tokenizer, speech_structure=speech_structure
            )
            for input_list in inputs
        ]

    @timer("llm inference")
    @torch.inference_mode()
    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens: int = 300,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[ModelResponse]:
        """
        Generates a list of texts in response to the given input.

        Args:
            inputs: A list of list of model inputs. Each ModelInput corresponds roughly to one command,
                a list of ModelInputs corresponds to a single debate (or entry in a batch), and so the
                list of lists is basically a batch of debates.
            max_new_tokens: the maximum number of new tokens to generate.
            speech_structure: the format that the answer is expected to be in. Option includes "open-ended"
                (which is just free text), and "decision" (which means a boolean is expected)
            num_return_sequences: the number of responses that the model is expected to generate. If a batch
                size of >1 is passed in, then this value will be overridden by the batch size (so you cannot
                have both num_return_sequences > 1 and len(inputs) > 1)

        Returns:
            A list of model responses, with one response for each entry in the batch (or for as many sequences
            are specified to be returned by num_return_sequences).

        Raises:
            Exception: Raises Exception if num_return_sequences > 1 and len(inputs) > 1
        """

        def validate():
            if num_return_sequences > 1 and len(inputs) > 1:
                raise Exception("You cannot have multiple return sequences and a batch size of >1")

        def get_string_log_prob(target_string: list[str], logits: torch.Tensor, batch_index: int) -> float:
            prob = 0
            for i, target in enumerate(self.tokenizer(target_string).input_ids[1:]):
                prob += F.log_softmax(logits[i][batch_index].squeeze())[target].item()
            return prob

        def normalize_log_probs(a_prob: float, b_prob: float) -> tuple[float, float]:
            exponentiated = [math.exp(logprob) for logprob in [a_prob, b_prob]]
            return exponentiated[0] / sum(exponentiated), exponentiated[1] / sum(exponentiated)

        def create_new_generation_config():
            config_to_use = copy.deepcopy(self.generation_config)
            config_to_use.num_return_sequences = num_return_sequences
            return config_to_use

        def generate_output(input_strs: list[str]):
            sequences = []
            logits = []
            input_lengths = []
            minibatches = [
                input_strs[i : i + self.max_mini_batch_size]
                for i in range(0, len(input_strs), self.max_mini_batch_size)
            ]
            for minibatch in minibatches:
                inputs = self.tokenizer(minibatch, return_tensors="pt", padding=True).to(device)
                outputs = self.model.generate(**inputs, generation_config=create_new_generation_config())
                mini_sequences = outputs.sequences if not isinstance(self.model, LLModuleWithLinearProbe) else outputs
                sequences += [row for row in mini_sequences]
                logits += [row for row in outputs.scores] if hasattr(outputs, "scores") else []
                input_lengths += [elem for elem in (inputs.input_ids != self.tokenizer.pad_token_id).sum(axis=1)]
            return (
                sequences,
                torch.stack(input_lengths),
                torch.stack(logits) if logits else None,
            )

        validate()
        self.model.eval()
        device = get_device()
        input_strs = self.generate_input_strs(inputs=inputs, speech_structure=speech_structure)
        sequences, input_lengths, logits = generate_output(input_strs=input_strs)

        decoded_outputs = []
        for i, _ in enumerate(sequences):
            if self.is_debater or speech_structure != SpeechStructure.DECISION:
                prompt_tokens = sequences[i][: input_lengths[i]]
                response_tokens = sequences[i][input_lengths[i] :]
                decoded = self.tokenizer.decode(response_tokens, skip_special_tokens=True)
                new_tokens = decoded.split(constants.INSTRUCTION_SUFFIX)[-1]
                decoded_outputs.append(
                    ModelResponse(
                        speech=string_utils.clean_string(new_tokens),
                        prompt=input_strs[i],
                        prompt_tokens=prompt_tokens.squeeze().tolist(),
                        response_tokens=response_tokens.squeeze().tolist(),
                    )
                )
            else:
                internal_representations = []
                if isinstance(self.model, LLModuleWithLinearProbe):
                    outputs = None
                    (a_score, b_score), internal_representations = outputs[i]
                else:
                    a_score = get_string_log_prob(constants.DEFAULT_DEBATER_A_NAME, logits, i)
                    b_score = get_string_log_prob(constants.DEFAULT_DEBATER_B_NAME, logits, i)

                normalized_a_score, normalized_b_score = normalize_log_probs(a_score, b_score)
                decoded_outputs.append(
                    ModelResponse(
                        decision=(
                            constants.DEFAULT_DEBATER_A_NAME if a_score > b_score else constants.DEFAULT_DEBATER_B_NAME
                        ),
                        probabilistic_decision={
                            constants.DEFAULT_DEBATER_A_NAME: normalized_a_score,
                            constants.DEFAULT_DEBATER_B_NAME: normalized_b_score,
                        },
                        prompt=input_strs[i],
                        internal_representations=internal_representations if internal_representations else None,
                    )
                )

                self.logger.info(f"Scores: A {normalized_a_score} - B {normalized_b_score}")

        return decoded_outputs

    def copy(self, alias: str, is_debater: bool | None = None, nucleus: bool = False) -> LLModel:
        """Generates a deepcopy of this model"""
        copy = LLModel(
            alias=alias,
            is_debater=self.is_debater if is_debater is None else is_debater,
            nucleus=nucleus,
            generation_params=self.generation_params,
        )
        copy.is_debater = self.is_debater if is_debater is None else is_debater
        copy.tokenizer = self.tokenizer
        copy.model = self.model
        copy.generation_config = self.generation_config
        return copy


class LlamaModel(LLModel):
    INSTRUCTION_PREFIX = "instruction:"
    INSTRUCTION_SUFFIX = "output:"
    ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj"]
    MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
    TARGET_MODULES = ["k_proj", "v_proj", "down_proj"]

    def __init__(
        self,
        alias: str,
        file_path: str | None = None,
        is_debater: bool = True,
        nucleus: bool = True,
        probe_hyperparams: ProbeHyperparams | None = None,
        generation_params: GenerationParams | None = None,
        peft_base_model: str | None = None,
    ):
        super().__init__(
            alias=alias,
            file_path=file_path,
            is_debater=is_debater,
            nucleus=nucleus,
            instruction_prefix="instruction:",
            instruction_suffix="output:",
            requires_file_path=True,
            probe_hyperparams=probe_hyperparams,
            max_mini_batch_size=1,
            generation_params=generation_params,
            peft_base_model=peft_base_model,
        )

        if self.model:
            self.model.config.max_position_embeddings = constants.MAX_LENGTH

    def copy(self, alias: str, is_debater=None, nucleus: bool = False) -> LLModel:
        """Generates a deepcopy of this model"""
        copy = LlamaModel(
            alias=alias,
            is_debater=self.is_debater if is_debater is None else is_debater,
            nucleus=nucleus,
            generation_params=self.generation_params,
        )
        copy.is_debater = self.is_debater if is_debater is None else is_debater
        copy.tokenizer = self.tokenizer
        copy.model = self.model
        copy.generation_config = self.generation_config
        return copy


class MistralModel(LLModel):
    INSTRUCTION_PREFIX = "[INST]"
    INSTRUCTION_SUFFIX = "[/INST]"
    ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj"]
    MLP_MODULES = []
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]
    LINEAR_IDXS = [31, 16]

    def __init__(
        self,
        alias: str,
        file_path: str | None = None,
        is_debater: bool = True,
        nucleus: bool = True,
        probe_hyperparams: ProbeHyperparams | None = None,
        generation_params: GenerationParams | None = None,
        peft_base_model: str | None = None,
    ):
        super().__init__(
            alias=alias,
            file_path=file_path,
            is_debater=is_debater,
            nucleus=nucleus,
            instruction_prefix="[INST]",
            instruction_suffix="[/INST]",
            requires_file_path=True,
            probe_hyperparams=probe_hyperparams,
            max_mini_batch_size=1,
            generation_params=generation_params,
            peft_base_model=peft_base_model,
        )

        if self.model:
            self.model.config.sliding_window = constants.MAX_LENGTH

    def copy(self, alias: str, is_debater: bool | None = None, nucleus: bool = False) -> LLModel:
        """Generates a deepcopy of this model"""
        copy = MistralModel(
            alias=alias,
            is_debater=self.is_debater if is_debater is None else is_debater,
            nucleus=nucleus,
            generation_params=self.generation_params,
        )
        copy.is_debater = self.is_debater if is_debater is None else is_debater
        copy.tokenizer = self.tokenizer
        copy.model = self.model
        copy.generation_config = self.generation_config
        return copy


class Llama3Model(LLModel):
    INSTRUCTION_PREFIX = ""
    INSTRUCTION_SUFFIX = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    ATTENTION_MODULES = ["q_proj", "k_proj", "v_proj"]
    MLP_MODULES = ["gate_proj", "up_proj", "down_proj"]
    TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    LINEAR_IDXS = [31, 16]
    QUANTIZE = False

    def __init__(
        self,
        alias: str,
        file_path: str | None = None,
        is_debater: bool = True,
        nucleus: bool = True,
        probe_hyperparams: ProbeHyperparams | None = None,
        generation_params: GenerationParams | None = None,
        peft_base_model: str | None = None,
    ):
        if generation_params is None:
            generation_params = GenerationParams()

        super().__init__(
            alias=alias,
            file_path=file_path,
            is_debater=is_debater,
            nucleus=nucleus,
            instruction_prefix="",
            instruction_suffix="",
            requires_file_path=True,
            probe_hyperparams=probe_hyperparams,
            max_mini_batch_size=1,
            quantize=False,
            generation_params=generation_params,
            peft_base_model=peft_base_model,
        )

    def copy(self, alias: str, is_debater=None, nucleus: bool = False) -> LLModel:
        """Generates a deepcopy of this model"""
        copy = Llama3Model(
            alias=alias,
            is_debater=self.is_debater if is_debater is None else is_debater,
            nucleus=nucleus,
            generation_params=self.generation_params,
        )
        copy.is_debater = self.is_debater if is_debater is None else is_debater
        copy.tokenizer = self.tokenizer
        copy.model = self.model
        copy.generation_config = self.generation_config
        return copy

    def create_default_generation_config(self, is_debater: bool = True, generation_params=None) -> GenerationConfig:
        """Creates a default generation config so that the model can generate text"""
        if generation_params is None:
            generation_params = GenerationParams()

        generation_config = super().create_default_generation_config(
            is_debater=is_debater, generation_params=generation_params
        )
        generation_config.eos_token_id = [
            self.tokenizer.eos_token_id,
            self.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
        return generation_config


class StubLLModel(LLModel):
    def __init__(
        self,
        alias: str,
        file_path=None,
        is_debater: bool = True,
        nucleus: bool = True,
        generation_params=None,
    ):
        if generation_params is None:
            generation_params = GenerationParams()
        super().__init__(
            alias=alias,
            file_path=file_path,
            is_debater=is_debater,
            nucleus=nucleus,
            instruction_prefix="",
            instruction_suffix="",
            requires_file_path=False,
            generation_params=generation_params,
        )

    def copy(self, alias: str, is_debater=None, nucleus: bool = False) -> LLModel:
        """Generates a deepcopy of this model"""
        return StubLLModel(
            alias=alias,
            is_debater=self.is_debater if is_debater is None else is_debater,
            nucleus=nucleus,
            generation_params=self.generation_params,
        )

    def instantiate_tokenizer_and_hf_model(
        self, file_path: str, **kwargs
    ) -> tuple[AutoTokenizer, AutoModelForCausalLM]:
        """Constructs the stub tokenizer and stub model"""
        return TokenizerStub(), ModelStub()


class LLModuleWithLinearProbe(nn.Module):
    def __init__(self, base_model: LLModel, linear_idxs: list[int] | None = None, file_path: str = ""):
        super().__init__()
        self.linear_idxs = linear_idxs or [-1]
        self.base_model = base_model.model.to(get_device())
        self.base_model.eval()
        self.config = self.base_model.config
        self.probe = LLModuleWithLinearProbe.instantiate_probe(
            file_path=file_path,
            linear_idxs=self.linear_idxs,
            hidden_size=self.base_model.config.hidden_size,
        )
        # self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid(dim=1)

    @classmethod
    def instantiate_probe(cls, file_path: str, linear_idxs: list[int], hidden_size: int) -> nn.Module:
        device = get_device()
        probe = nn.Linear(in_features=hidden_size * len(linear_idxs), out_features=1)
        if file_path:
            probe.load_state_dict(torch.load(file_path))
        else:
            raise Exception(f"File path ({file_path}) not loaded")
        return probe.to(device)

    def encode_representation(self, representation: torch.tensor) -> str:
        buffer = io.BytesIO()
        torch.save(representation, buffer)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode("utf-8")

    def generate(self, input_ids: torch.tensor, **kwargs) -> list[tuple(tuple(float, float), torch.tensor)]:
        return self.forward(input_ids=input_ids)

    def forward(self, input_ids=None) -> list[tuple(tuple(float, float), torch.tensor)]:
        batch_size = input_ids.shape[0]

        base_model_output = self.base_model(input_ids=input_ids.to("cuda"), output_hidden_states=True)

        hidden_states = [[] for i in range(batch_size)]
        for _, layer in enumerate(base_model_output.hidden_states):
            for j in range(batch_size):
                hidden_states[j].append(layer[j, -1, :])

        input_vecs = torch.stack(
            [torch.cat([hidden_states[i][idx] for idx in self.linear_idxs], dim=0) for i in range(batch_size)]
        )

        unnormalized_outputs = self.probe(input_vecs.float())
        # outputs = self.softmax(unnormalized_outputs)
        a_odds = self.sigmoid(unnormalized_outputs)
        outputs = [a_odds, 1 - a_odds]
        reformatted_outputs = [(output[0].item(), output[1].item()) for output in outputs]
        encoded_hidden_states = [self.encode_representation(hs) for hs in hidden_states]

        return [(ro, ehs) for ro, ehs in zip(reformatted_outputs, encoded_hidden_states)]

    def parameters(self):
        return self.probe.parameters()


class LLMType(Enum):
    LLAMA = auto()
    MISTRAL = auto()
    OPENAI = auto()
    STUB_LLM = auto()
    LLAMA3 = auto()

    def get_llm_class(self):
        if self == LLMType.LLAMA:
            return LlamaModel
        elif self == LLMType.MISTRAL:
            return MistralModel
        elif self == LLMType.STUB_LLM:
            return StubLLModel
        elif self == LLMType.OPENAI:
            return OpenAIModel
        elif self == LLMType.LLAMA3:
            return Llama3Model
        else:
            raise Exception(f"Model type {self} not recognized")


@dataclass
class ModelConfigStub:
    max_position_embeddings: int = 0


class TokenizerOutputStub:
    def __init__(self, input_ids: torch.tensor):
        self.input_ids = input_ids
        self.__data = {"input_ids": self.input_ids}

    def __iter__(self):
        return iter(self.__data)

    def keys(self):
        return self.__data.keys()

    def __getitem__(self, key):
        return self.__data[key]

    def to(self, device: str):
        return self


class TokenizerStub:
    def __init__(self):
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.alphabet = "abcdefghijklmnopqrstuvwxyz"
        self.chat_template = None

    def __call__(self, inputs: list[str], **kwargs) -> dict[str, torch.tensor]:
        batch_size = len(inputs)
        max_sequence_length = max(len(seq) for seq in inputs)
        return TokenizerOutputStub(input_ids=torch.tensor(np.random.randint(0, 100, [batch_size, max_sequence_length])))

    def encode(self, input_string: str | list[str], **kwargs):
        if not isinstance(input_string, str) or not isinstance(input_string, list):
            return input_string

        if isinstance(input_string, str):
            input_string = [input_string]
        input_ids = self(input_string).input_ids
        if len(input_string) == 1:
            return input_ids[0, :]
        return input_ids

    def decode(self, tokens: torch.tensor, **kwargs) -> str | list[str]:
        if len(tokens.shape) == 1:
            batch_size = 1
            sequence_length = tokens.shape[0]
        else:
            batch_size, sequence_length = tokens.shape
        outputs = [
            " ".join(["".join(random.choices(self.alphabet, k=random.randrange(1, 8))) for i in range(sequence_length)])
            for _ in range(batch_size)
        ]
        if len(outputs) == 1:
            return outputs[0]
        else:
            return outputs


@dataclass
class ModelOutputStub:
    sequences: Any  # should be a torch tensor


class ModelStub:
    def __init__(self):
        self.config = ModelConfigStub()

    def train(self):
        pass

    def eval(self):
        pass

    def generate(self, input_ids: torch.tensor, generation_config: GenerationConfig, **kwargs):
        return self(input_ids=input_ids, generation_config=generation_config, **kwargs)

    def __call__(self, input_ids: torch.tensor, generation_config: GenerationConfig, **kwargs):
        batch_size, sequence_length = input_ids.shape
        output_sequence_length = sequence_length + generation_config.max_new_tokens
        return ModelOutputStub(sequences=torch.tensor(np.random.randint(0, 100, [batch_size, output_sequence_length])))
