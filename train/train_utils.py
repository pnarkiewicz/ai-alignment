from enum import Enum
from typing import Any, Optional

import torch
import yaml
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    PeftType,
    PromptTuningConfig,
    PromptTuningInit,
    TaskType,
)
from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import AutoModelForCausalLMWithValueHead

import utils.constants as constants
from data import DatasetConfig, RawDataset, loader_utils
from debate import ScratchpadConfig, SpeechFormatStructure
from models import LLModel, LLMType, ModelStub, TokenizerStub
from models.arbitrary_attribute_model import ArbitraryAttributeModel
from models.hf_wrapper_judge import HFWrapperJudge
from models.deterministic_model import DeterministicModel
from models.openai_model import OpenAIModel
from models.random_model import RandomModel
from prompts import PromptLoadingConfig


class TrainingTarget(Enum):
    DEBATER = 1
    JUDGE = 2


class LoggingAndSavingConfig(BaseModel):
    logging_steps: int
    output_dir: str
    merge_output_dir: Optional[str] = None


class TargetModule(Enum):
    ALL = 0
    ATTENTION = 1
    MLP = 2


class TrainingHyperParameterConfig(BaseModel):
    num_train_epochs: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int = 1
    optim: Optional[str] = None
    learning_rate: float
    max_grad_norm: float = float("inf")
    warmup_ratio: float = 0
    lr_scheduler_type: str = ""
    peft_type: PeftType | str = ""
    steps: Optional[int] = None
    lora_rank: int = 64
    kl_penalty: float = 0.1
    target_module: TargetModule | str = TargetModule.ATTENTION
    supplemental: Optional[dict[str, Any]] = {}

    @field_validator("target_module", mode="before")
    @classmethod
    def validate_training_target(cls, target_module: str | TargetModule):
        if isinstance(target_module, str):
            return TargetModule[target_module.upper()]
        return target_module

    @field_validator("peft_type", mode="before")
    @classmethod
    def validate_peft_type(cls, peft_type: str | PeftType):
        if peft_type and isinstance(peft_type, str):
            return PeftType[peft_type.upper()]
        return peft_type


class TrainingConfig(BaseModel):
    model_name: str
    reference_model_name: Optional[str] = None
    llm_type: str = "llama"
    prompt_config: PromptLoadingConfig = PromptLoadingConfig()
    logging_and_saving_config: Optional[LoggingAndSavingConfig] = None
    training_hyperparameters: Optional[TrainingHyperParameterConfig] = None
    target: TrainingTarget = TrainingTarget.DEBATER
    dataset: DatasetConfig | list[DatasetConfig]
    opening_speeches_only: bool = False
    requires_token: bool = False
    max_length: int = constants.MAX_LENGTH
    scratchpad_config: ScratchpadConfig = ScratchpadConfig()
    speech_structure: SpeechFormatStructure | list[SpeechFormatStructure] = [SpeechFormatStructure.DEFAULT_DEBATE]
    tokenizer_file_path: Optional[str] = None
    model_config = ConfigDict(protected_namespaces=("protect_me_", "also_protect_"))

    @field_validator("speech_structure", mode="before")
    @classmethod
    def validate_speech_structure(
        cls,
        speech_structure: (str | SpeechFormatStructure | list[str] | list[SpeechFormatStructure]),
    ):
        if not isinstance(speech_structure, list):
            speech_structure = [speech_structure]

        new_speech_structure = []
        for s in speech_structure:
            if isinstance(s, str):
                new_speech_structure.append(SpeechFormatStructure[s.upper()])
            else:
                new_speech_structure.append(s)

        return new_speech_structure

    @field_validator("target", mode="before")
    @classmethod
    def validate_training_target(cls, target: str | TrainingTarget):
        if isinstance(target, str):
            return TrainingTarget[target.upper()]
        return target

    @field_validator("target", mode="before")
    @classmethod
    def validate_dataset(cls, dataset: DatasetConfig | list[DatasetConfig]):
        if isinstance(dataset, DatasetConfig):
            return [dataset]
        return dataset

    @model_validator(mode="after")
    @classmethod
    def ensure_speech_structure_matches_datasets(cls, values):
        if len(values.speech_structure) > 1 and len(values.speech_structure) != len(values.dataset):
            raise ValueError("We could not match speech structures to each dataset")
        return values


class TrainUtils:
    @classmethod
    def create_datasets(cls, config: TrainingConfig, deduplicate: bool = False, **kwargs) -> list[RawDataset]:
        """
        Constructs a dataset that will later be converted into a training dataset.

        Params:
            config: the configuration containing the prompt text and training hyperparameters
            deduplicate: whether only one example from each prompt should be used

        Returns:
            dataset: a list of dataset objects that can later be used as training datasets
        """
        dataset_configs = config.dataset
        if isinstance(dataset_configs, DatasetConfig):
            dataset_configs = [dataset_configs]

        datasets = []
        for dataset_config in filter(lambda x: x.dataset_type.is_instantiable, dataset_configs):
            loader_cls = loader_utils.get_loader_type(dataset_config.dataset_type)
            dataset = loader_cls.load(
                full_dataset_filepath=dataset_config.full_dataset_file_path,
                train_filepath=dataset_config.train_file_path,
                val_filepath=dataset_config.val_file_path,
                test_filepath=dataset_config.test_file_path,
                supplemental_file_paths=dataset_config.supplemental_file_paths,
                deduplicate=deduplicate,
                flip_sides=dataset_config.flip_sides,
                **kwargs,
            )
            datasets.append(dataset)
        return datasets

    @classmethod
    def parse_config(cls, config_name: Optional[str], config_filepath: str) -> TrainingConfig:
        """Loads a yaml file and converts it into a training configuration"""
        with open(config_filepath) as f:
            loaded_yaml = yaml.safe_load(f)
        config_name = config_name or [key for key in config_name][0]
        return TrainingConfig(**loaded_yaml[config_name])

    @classmethod
    def get_peft_config(cls, config: TrainingConfig) -> Optional[PeftConfig]:
        """Gets the configuration from parameter efficient fine tuning"""
        peft_type = config.training_hyperparameters.peft_type
        if not peft_type:
            return None
        llm_class = TrainUtils.get_llm_class(config=config)

        targets = llm_class.TARGET_MODULES
        if config.training_hyperparameters.target_module:
            if config.training_hyperparameters.target_module == TargetModule.ATTENTION:
                targets = llm_class.ATTENTION_MODULES
            elif config.training_hyperparameters.target_module == TargetModule.MLP:
                targets = llm_class.MLP_MODULES
            elif config.training_hyperparameters.target_module == TargetModule.ALL:
                targets = llm_class.ATTENTION_MODULES + llm_class.MLP_MODULES

        if peft_type == PeftType.LORA:
            return LoraConfig(
                lora_alpha=16,
                lora_dropout=0.0,
                r=config.training_hyperparameters.lora_rank,
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=targets,
                init_lora_weights="gaussian",
            )
        elif peft_type == PeftType.PROMPT_TUNING:
            return PromptTuningConfig(
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=16,
                prompt_tuning_init_text="Now give your speech:",
                tokenizer_name_or_path=config.model_name,
                task_type=TaskType.CAUSAL_LM,
            )

    @classmethod
    def load_training_model(
        cls,
        config: TrainingConfig,
        is_local: bool = False,
        requires_value_head: bool = False,
        load_as_peft_model: bool = False,
    ) -> AutoModelForCausalLM:
        """
        Loads a model using the specified configuration.

        Params:
            config: the configuration covering the training hyperparameters
            is_local: whether it's being run on a cpu
            requires_value_head: whether we need to wrap the model with a layer that generates scalar values.
                (Only used for PPO training for now)

        Returns:
            model: a model loaded from huggingface
        """

        if not is_local:
            peft_config = TrainUtils.get_peft_config(config=config)
            llm_cls = TrainUtils.get_llm_class(config)
            model_name = config.model_name if not load_as_peft_model else config.reference_model_name
            model = LLModel.instantiate_hf_model(
                file_path=model_name,
                requires_token=config.requires_token,
                use_cache=False,
                quantize=llm_cls.QUANTIZE,
            )

            if load_as_peft_model:
                adapter_path = config.model_name
                model = PeftModel.from_pretrained(
                    model=model,
                    model_id=adapter_path,
                    adapter_name="default",
                    is_trainable=True,
                )

            if requires_value_head:
                use_flash_attention = True
                try:
                    import flash_attn
                except ImportError:
                    use_flash_attention = False

                quantization_config = None
                if llm_cls.QUANTIZE and torch.cuda.is_available():
                    quantization_config = LLModel.get_bnb_config()

                model = AutoModelForCausalLMWithValueHead.from_pretrained(
                    pretrained_model_name_or_path=model,
                    trust_remote_code=True,
                    use_flash_attention_2=use_flash_attention,
                    use_cache=False,
                    peft_config=peft_config,
                    quantization_config=quantization_config,
                    torch_dtype=None if llm_cls.QUANTIZE else torch.bfloat16,
                )

            model.config.max_position_embeddings = config.max_length
            model.config.sliding_window = config.max_length
            model.config.max_position_embeddings = config.max_length
            return model
        else:
            return ModelStub()

    @classmethod
    def load_judge_model(cls, config: TrainingConfig, is_local: bool = False) -> AutoModelForCausalLM:
        """Loads the judge model"""

        DEFAULT_JUDGE_ALIAS = "default-judge"
        supplemental = config.training_hyperparameters.supplemental or {}
        judge_type = supplemental.get("judge_type", "openai")

        if judge_type == 'hf_wrapper':
            return HFWrapperJudge(
                alias=DEFAULT_JUDGE_ALIAS,
                is_debater=False,
                feature=supplemental.get("judge_model"),
            )
        elif judge_type == "arbitrary_attribute":
            return ArbitraryAttributeModel(
                alias=DEFAULT_JUDGE_ALIAS,
                is_debater=False,
                feature=supplemental.get("feature", "l"),
            )

        if judge_type == "deterministic":  # Currently, DEFAULT_DEBATER_A_NAME is hardcoded as the winner
            return DeterministicModel(
                alias=DEFAULT_JUDGE_ALIAS,
                is_debater=False,
            )

        if is_local:
            return RandomModel(alias=DEFAULT_JUDGE_ALIAS, is_debater=False)

        else:
            return OpenAIModel(
                alias=DEFAULT_JUDGE_ALIAS,
                is_debater=False,
                endpoint="ft:gpt-4-0613:nyu-arg::90NW3Tbx",
            )

    @classmethod
    def get_tokenizer(cls, config: TrainingConfig, is_local: bool = False) -> AutoTokenizer:
        """Gets the tokenizer associated with the specified model"""
        if is_local or config.model_name == "stub_model":
            return TokenizerStub()
        return LLModel.instantiate_tokenizer(file_path=config.model_name, requires_token=config.requires_token)

    @classmethod
    def get_llm_class(cls, config: TrainingConfig):
        return LLMType[config.llm_type.upper()].get_llm_class()
