from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import torch
from pydantic import BaseModel, field_validator


class SplitType(Enum):
    TRAIN = 1
    VAL = 2
    TEST = 3


class DatasetType(Enum):
    QUALITY = (1, True)
    QUALITY_DEBATES = (2, True)
    JUDGE_PREFERENCES = (3, True)
    ANNOTATED_QUALITY_DEBATES = (4, True)
    SCRATCHPAD_QUALITY_DEBATES = (5, True)
    QUOTE_RELEVANCE = (6, True)
    JUDGING_PROBE = (7, True)
    QUALITY_CONSULTANCY = (8, True)
    CORRECTNESS_JUDGE_PREFERENCES = (9, True)
    EXTERNAL_HUGGINGFACE = (10, False)
    TRUTHFUL = (11, True)
    KHUN_POKER = (12, True)  # TODO: not implemented yet
    GSM8K = (13, True)

    def __init__(self, idx: int, is_instantiable: bool):
        self.id = idx
        self.is_instantiable = is_instantiable


class DatasetConfig(BaseModel):
    dataset_type: DatasetType
    full_dataset_file_path: Optional[str | list[str]] = None
    train_file_path: Optional[str] = None
    val_file_path: Optional[str] = None
    test_file_path: Optional[str] = None
    supplemental_file_paths: dict[str, str | list[str]] = {}
    split_type: SplitType = SplitType.TRAIN
    combine_train_and_val: bool = False
    flip_sides: bool = False
    shuffle_deterministically: bool = False

    @field_validator("split_type", mode="before")
    @classmethod
    def validate_tournament_type(cls, split_type: str):
        return SplitType[split_type.upper()]

    @field_validator("dataset_type", mode="before")
    @classmethod
    def validate_dataset_type(cls, dataset_type: str):
        return DatasetType[dataset_type.upper()]


class SpeakerType(Enum):
    DEBATER = 1
    JUDGE = 2


class AnnotationTag(Enum):
    QUOTE = 0
    SUMMARY = 1
    REFUTATION = 2
    ANALYSIS = 3
    REPLY = 4
    FLOURISH = 5
    FRAMING = 6
    STATEMENT = 7
    LOGIC = 8
    Q_CONTEXT = 9
    POSITION = 10
    OOB_QUOTE = 11
    PROMISE = 12


class AnnotationBracket(Enum):
    HIGH = 1
    LOW = 2
    NEUTRAL = 3


class AnnotationData(BaseModel):
    percents: Optional[dict[AnnotationTag | str, float]] = None
    percentiles: Optional[dict[AnnotationTag | str, float]] = None


class SpeechData(BaseModel):
    text: str
    position: int
    speaker_type: SpeakerType
    scratchpad: Optional[str] = None
    annotation: Optional[AnnotationData] = None
    probabilities: Optional[tuple[float, float]] = None


class DataRow(BaseModel):
    background_text: str
    question: Optional[str] = None
    positions: Optional[tuple[str, str]] = None
    speeches: Optional[list[SpeechData]] = None
    correct_index: Optional[int] = None
    debate_id: Optional[str] = None
    story_title: Optional[str] = None


class JudgePreferenceDataRow(BaseModel):
    prompt: str
    chosen: str
    rejected: str
    preference: float = 1.0


@dataclass
class JudgingProbeDataRow:
    internal_representation: torch.tensor
    target: torch.tensor


class RawDataset(ABC):
    def __init__(self, dataset_type: DatasetType):
        self.dataset_type = dataset_type

    @abstractmethod
    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[tuple[str, Any]]:
        """Fetches all the data for a given split of the data"""
        pass

    @abstractmethod
    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[tuple[str, Any]]:
        """Gets a subset of the data"""
        pass

    @abstractmethod
    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> DataRow:
        """Returns an individual row at the specified index"""
        pass

    def get_dataset_type(self):
        """Gets the name of the dataset"""
        return self.dataset_type


class RawDataLoader(ABC):
    @classmethod
    @abstractmethod
    def load(
        cls,
        full_dataset_filepath: str | None = None,
        train_filepath: str | None = None,
        validation_filepath: str | None = None,
        test_filepath: str | None = None,
        supplemental_file_paths: str | None = None,
        combine_train_and_val: bool = False,
    ) -> RawDataset:
        """Constructs a dataset"""
        pass
