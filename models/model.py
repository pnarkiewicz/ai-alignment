from __future__ import annotations

from abc import ABC
from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator, validator

import utils.constants as constants
from prompts import RoleType
from utils.constants import DEBUG


class ModelType(Enum):
    RANDOM = 1
    LLAMA = 2
    DETERMINISTIC = 3
    OPENAI = 4
    OFFLINE = 5
    HUMAN = 6
    MISTRAL = 7
    STUB_LLM = 8
    ARBITRARY_ATTRIBUTE = 9
    ANTHROPIC = 10
    LLAMA3 = 11
    REPETITIVE = 12


class BestOfNConfig(BaseModel):
    n: int
    opponent_n: int = 0
    maxmin: bool = False
    recompute: bool = False


class GenerationParams(BaseModel):
    max_new_tokens: int = 300
    temperature: float = 0.7 if DEBUG else 0.5  # TODO: parametrise this
    top_p: float = 1.0
    repetition_penalty: float = 1.2
    do_sample: bool = True
    use_generation_penalties: bool = False


class ModelInput(BaseModel):
    role: RoleType
    content: str


class ModelResponse(BaseModel):
    speech: str = ""
    decision: Literal[constants.DEFAULT_DEBATER_A_NAME, constants.DEFAULT_DEBATER_B_NAME, ""] = ""
    probabilistic_decision: Optional[dict[str, float]] = None
    preference: Optional[float] = None
    rejected_responses: list[ModelResponse] = []
    bon_opposing_model_responses: list[ModelResponse] = []
    bon_probabilistic_preferences: list[float] = []
    internal_representations: Optional[str] = ""
    response_tokens: list[int] = []
    prompt_tokens: list[int] = []
    prompt: str = ""
    failed: bool = False

    @validator("probabilistic_decision")
    def check_keys(cls, v):
        if v:
            if constants.DEFAULT_DEBATER_A_NAME not in v:
                raise ValueError(f"Probabilistic decision is missing required key: {constants.DEFAULT_DEBATER_A_NAME}")
            if constants.DEFAULT_DEBATER_B_NAME not in v:
                raise ValueError(f"Probabilistic decision is missing required key: {constants.DEFAULT_DEBATER_B_NAME}")
            if len(v) > 2:
                all_keys = ", ".join(v.keys())
                raise ValueError(f"There are too many keys in the probabilistic decision map. Keys: {all_keys}")

            eps = 0.001
            total_prob = sum(v.values())
            if total_prob < 1 - eps or total_prob > 1 + eps:
                raise ValueError(f"Total probability does not sum to 1 -- it sums to {total_prob}. Map is {v}")

        return v


class ProbeHyperparams(BaseModel):
    file_path: str = ""
    hidden_size: Optional[int] = None
    linear_idxs: list[int] = [-1]


class ModelSettings(BaseModel):
    model_type: str | ModelType = ModelType.RANDOM
    model_file_path: Optional[str] = None
    alias: str
    override_prompt: Optional[str] = None
    nucleus: bool = True
    is_human: bool = False
    offline_file_path: Optional[str] = None
    served: bool = False
    probe_hyperparams: Optional[ProbeHyperparams] = None
    require_quote_validation: bool = True
    generation_params: GenerationParams = GenerationParams()
    peft_base_model: Optional[str] = None

    @model_validator(mode="before")
    def verify_custom_settings(cls, values):
        existence_count = sum([values.get("is_human", False), values.get("served", False)]) + (
            1 if values.get("offline_file_path", None) else 0
        )
        if existence_count > 1:
            raise ValueError(
                "One cannot set more than one of is_human, served, or offline_file_path to non-null and true"
            )
        return values

    model_config = ConfigDict(protected_namespaces=("protect_me_", "also_protect_"))

    @field_validator("alias", mode="before")
    @classmethod
    def validate_alias(cls, alias: str | int):
        return str(alias)

    @field_validator("model_type", mode="before")
    @classmethod
    def validate_model_type(cls, model_type: str | ModelType):
        if isinstance(model_type, str):
            return ModelType[model_type.upper()]
        return model_type


class SpeechStructure(Enum):
    OPEN_ENDED = 1
    DECISION = 2


class Model(ABC):
    def __init__(self, alias: str, is_debater: bool = False):
        self.alias = alias
        self.is_debater = is_debater

    def predict(self, inputs: list[list[ModelInput]], max_new_tokens: 250, **kwargs) -> ModelResponse:
        pass

    def copy(self, is_debater: Optional[bool] = None, **kwargs) -> Model:
        return self

    def can_merge(self, other: Model) -> bool:
        return other == self

    def merge(self, other: Model) -> Model:
        if self.can_merge(other):
            return self
        raise Exception("Cannot merge across models")
