import os
import random
from typing import Any, Optional

from datasets import load_dataset

from data.annotated_quality_debates_loader import AnnotatedQualityDebatesLoader
from data.dataset import DataRow, DatasetType, RawDataLoader, RawDataset, SplitType
from data.judge_preferences_loader import CorrectnessJudgePreferencesLoader, JudgePreferencesLoader
from data.quality_debates_loader import QualityConsultancyLoader, QualityDebatesLoader, QualityTranscriptsLoader
from data.quality_judging_loader import QualityJudgingLoader
from data.quality_loader import QualityLoader
from data.gsm8k_loader import GSM8KLoader
from data.quote_relevance_loader import QuoteRelevanceLoader
from data.scratchpad_quality_debates_loader import ScratchpadQualityDebatesLoader
from utils import constants

"""
Most of the methods are taken from the QualityDataset class. Some might be not used.
"""

class TruthfulDataset(RawDataset):
    def __init__(
        self,
        train_data: list[dict[str, Any]],
        val_data: list[dict[str, Any]],
        test_data: list[dict[str, Any]],
        override_type: Optional[DatasetType] = None,
        allow_multiple_positions_per_question: bool = False,
        dedupe_dataset=None,
        flip_sides: bool = False,
        shuffle_deterministically: bool = False,
    ):
        """
        Dataset where each row contains a question and positions from the Quality dataset

        Params:
            train_data: list of training data loaded from quality jsonl file
            val_data: list of validation data loaded from quality jsonl file
            test_data: list of testing data loaded from quality jsonl file
            override_type: if this is being used as a parent class, this is the dataset type of the child
            allow_multiple_positions_per_question: many quality questions have more than two answers. By default,
                we will select the best distractor If this parameter is set to true, we will create
                a separate row for every single combination of positions
            dedupe_dataset: The dataset to dedupe from. This is used because the NYU Human Debate experiments used
                questions from the Quality dataset so if one trained on that data, then one needs to remove those
                rows from the validation set. Each entry is a dataset and a boolean indicating if one should dedupe
                all questions that share the same story (True) or not (False).
            flip_sides: Whether the ordering of the positions should be flipped (aka two rounds per question)
            shuffle_deterministically: Whether to use a fixed random seed for shuffling the dataset
        """
        super().__init__(override_type or DatasetType.QUALITY)
        if shuffle_deterministically:
            random.seed(a=123456789)
        self.allow_multiple_positions_per_question = allow_multiple_positions_per_question
        self.flip_sides = flip_sides
        self.data = {
            SplitType.TRAIN: self.__convert_batch_to_rows(train_data),
            SplitType.VAL: self.__convert_batch_to_rows(val_data),
            SplitType.TEST: self.__convert_batch_to_rows(test_data),
        }
        self.idxs = {SplitType.TRAIN: 0, SplitType.VAL: 0, SplitType.TEST: 0}
        if not self.data[SplitType.TEST]:  # Adding b/c Quality Test Set does not have gold labels
            self.__split_validation_and_test_sets()

        self.data[SplitType.TRAIN] = self.__reorder(self.data[SplitType.TRAIN])
        self.data[SplitType.VAL] = self.__reorder(self.data[SplitType.VAL])
        self.data[SplitType.TEST] = self.__reorder(self.data[SplitType.TEST])
        self.shuffle_deterministically = shuffle_deterministically

    def get_data(self, split: SplitType = SplitType.TRAIN) -> list[DataRow]:
        """Returns all the data for a given split"""
        if split not in self.data:
            raise ValueError(f"Split type {split} is not recognized. Only TRAIN, VAL, and TEST are recognized")
        return self.data[split]

    def get_batch(self, split: SplitType = SplitType.TRAIN, batch_size: int = 1) -> list[DataRow]:
        """Returns a subset of the data for a given split"""
        if batch_size < 1:
            raise ValueError(f"Batch size must be >= 1. Inputted batch size was {batch_size}")
        data_to_return = self.data[split][self.idxs[split] : min(self.idxs[split] + batch_size, len(self.data[split]))]
        self.idxs[split] = self.idxs[split] + batch_size if self.idxs[split] + batch_size < len(self.data[split]) else 0
        return [x for x in data_to_return]

    def get_example(self, split: SplitType = SplitType.TRAIN, idx: int = 0) -> DataRow:
        """Returns an individual row in the dataset"""
        return self.data[split][idx % len(self.data[split])]

    def __convert_batch_to_rows(self, batch: list[dict[str, Any]]):
        rows = []
        for entry in batch:
            correct_answer = entry["Best Answer"]
            incorrect_answer = entry["Incorrect Answers"].split(";")[0]
            first_correct = random.random() < 0.5
            data_row = DataRow(
                background_text=entry["Question"],
                question=entry["Question"],
                correct_index=0 if first_correct else 1,
                positions=(
                    correct_answer,
                    incorrect_answer,
                ) if first_correct else (
                    incorrect_answer,
                    correct_answer,
                ),
                story_title=entry["Question"],
                debate_id=entry["Question"],
            )
            rows.append(data_row)
        return rows

    def __split_validation_and_test_sets(self):
        second_half = self.data[SplitType.VAL][int(len(self.data[SplitType.VAL]) / 2) :]
        self.data[SplitType.VAL] = self.data[SplitType.VAL][0 : int(len(self.data[SplitType.VAL]) / 2)]
        val_stories = set([row.story_title for row in self.data[SplitType.VAL]])

        test_data = []
        for row in second_half:
            if row.story_title not in val_stories:
                test_data.append(row)
            else:
                self.data[SplitType.VAL].append(row)
        self.data[SplitType.TEST] = test_data

    def __reorder(self, rows: list[DataRow]) -> list[DataRow]:
        if len(rows) == 0:
            return rows

        random.shuffle(rows)
        story_to_rows = {}
        for row in rows:
            if row.story_title not in story_to_rows:
                story_to_rows[row.story_title] = []
            story_to_rows[row.story_title].append(row)

        final_order = []
        max_index = max([len(story_to_rows[row.story_title]) for row in rows])
        for index in range(max_index):
            for story in filter(lambda x: len(story_to_rows[x]) > index, story_to_rows):
                final_order.append(story_to_rows[story][index])
        return final_order


class TruthfulLoader(RawDataLoader):
    @classmethod
    def get_splits(
        cls,
        train_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
    ) -> tuple[list[dict]]:
        """Splits the data in train, val, and test sets"""
        DEFAULT_FILE_PATH = os.environ[constants.SRC_ROOT] + "data/datasets/TruthfulQA"

        dataset = load_dataset(DEFAULT_FILE_PATH)["train"]
        # TODO: It would be better to split to files so they are separated
        seed = 42
        train_data, tmp = dataset.train_test_split(test_size=0.2, seed=seed, shuffle=True).values()
        val_data, test_data = tmp.train_test_split(test_size=0.5, seed=seed, shuffle=True).values()
        return train_data, val_data, test_data

    @classmethod
    def load(
        cls,
        train_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
        allow_multiple_positions_per_question: bool = False,
        deduplicate_with_quality_debates: bool = True,
        supplemental_file_paths: Optional[dict[str, str]] = None,
        flip_sides: bool = False,
        shuffle_deterministically: bool = False,
        **kwargs,
    ) -> TruthfulDataset:
        """Constructs a QualityDataset"""
        train_split, val_split, test_split = TruthfulLoader.get_splits(
            train_filepath=train_filepath, val_filepath=val_filepath, test_filepath=test_filepath
        )

        dedupe_datasets = None
        if deduplicate_with_quality_debates:
            quality_debates_filepath = (supplemental_file_paths or {}).get(
                "quality_debates_file_path", QualityTranscriptsLoader.DEFAULT_FILE_PATH
            )
            quality_debates_dataset = QualityDebatesLoader.load(
                full_dataset_filepath=quality_debates_filepath, deduplicate=True
            )
            quality_consultancy_dataset = QualityConsultancyLoader.load(
                full_dataset_filepath=quality_debates_filepath, deduplicate=True
            )
            dedupe_datasets = [(quality_debates_dataset, True), (quality_consultancy_dataset, True)]

        return TruthfulDataset(
            train_data=train_split,
            val_data=val_split,
            test_data=test_split,
            allow_multiple_positions_per_question=allow_multiple_positions_per_question,
            dedupe_dataset=dedupe_datasets,
            flip_sides=flip_sides,
            shuffle_deterministically=shuffle_deterministically,
        )

    def __reorder(self, rows: list[DataRow]) -> list[DataRow]:
        if len(rows) == 0:
            return rows

        random.shuffle(rows)
        story_to_rows = {}
        for row in rows:
            if row.story_title not in story_to_rows:
                story_to_rows[row.story_title] = []
            story_to_rows[row.story_title].append(row)

        final_order = []
        max_index = max([len(story_to_rows[row.story_title]) for row in rows])
        for index in range(max_index):
            for story in filter(lambda x: len(story_to_rows[x]) > index, story_to_rows):
                final_order.append(story_to_rows[story][index])
        return final_order


class TruthfulLoader(RawDataLoader):
    @classmethod
    def get_splits(
        cls,
        train_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
    ) -> tuple[list[dict]]:
        """Splits the data in train, val, and test sets"""
        DEFAULT_FILE_PATH = os.environ[constants.SRC_ROOT] + "data/datasets/TruthfulQA"

        dataset = load_dataset(DEFAULT_FILE_PATH)["train"]
        # TODO: It would be better to split to files so they are separated
        seed = 42
        train_data, tmp = dataset.train_test_split(test_size=0.2, seed=seed, shuffle=True).values()
        val_data, test_data = tmp.train_test_split(test_size=0.5, seed=seed, shuffle=True).values()
        return train_data, val_data, test_data

    @classmethod
    def load(
        cls,
        train_filepath: Optional[str] = None,
        val_filepath: Optional[str] = None,
        test_filepath: Optional[str] = None,
        allow_multiple_positions_per_question: bool = False,
        deduplicate_with_quality_debates: bool = True,
        supplemental_file_paths: Optional[dict[str, str]] = None,
        flip_sides: bool = False,
        shuffle_deterministically: bool = False,
        **kwargs,
    ) -> TruthfulDataset:
        """Constructs a QualityDataset"""
        train_split, val_split, test_split = TruthfulLoader.get_splits(
            train_filepath=train_filepath, val_filepath=val_filepath, test_filepath=test_filepath
        )

        dedupe_datasets = None
        if deduplicate_with_quality_debates:
            quality_debates_filepath = (supplemental_file_paths or {}).get(
                "quality_debates_file_path", QualityTranscriptsLoader.DEFAULT_FILE_PATH
            )
            quality_debates_dataset = QualityDebatesLoader.load(
                full_dataset_filepath=quality_debates_filepath, deduplicate=True
            )
            quality_consultancy_dataset = QualityConsultancyLoader.load(
                full_dataset_filepath=quality_debates_filepath, deduplicate=True
            )
            dedupe_datasets = [(quality_debates_dataset, True), (quality_consultancy_dataset, True)]

        return TruthfulDataset(
            train_data=train_split,
            val_data=val_split,
            test_data=test_split,
            allow_multiple_positions_per_question=allow_multiple_positions_per_question,
            dedupe_dataset=dedupe_datasets,
            flip_sides=flip_sides,
            shuffle_deterministically=shuffle_deterministically,
        )


def get_loader_type(dataset_type: DatasetType) -> type[RawDataLoader]:
    """Returns the class associated with the inputted DatasetType"""
    if dataset_type == DatasetType.QUALITY:
        return QualityLoader
    elif dataset_type == DatasetType.QUALITY_DEBATES:
        return QualityDebatesLoader
    elif dataset_type == DatasetType.JUDGE_PREFERENCES:
        return JudgePreferencesLoader
    elif dataset_type == DatasetType.ANNOTATED_QUALITY_DEBATES:
        return AnnotatedQualityDebatesLoader
    elif dataset_type == DatasetType.SCRATCHPAD_QUALITY_DEBATES:
        return ScratchpadQualityDebatesLoader
    elif dataset_type == DatasetType.QUOTE_RELEVANCE:
        return QuoteRelevanceLoader
    elif dataset_type == DatasetType.JUDGING_PROBE:
        return QualityJudgingLoader
    elif dataset_type == DatasetType.QUALITY_CONSULTANCY:
        return QualityConsultancyLoader
    elif dataset_type == DatasetType.CORRECTNESS_JUDGE_PREFERENCES:
        return CorrectnessJudgePreferencesLoader
    elif dataset_type == DatasetType.TRUTHFUL:
        return TruthfulLoader
    elif dataset_type == DatasetType.GSM8K:
        return GSM8KLoader
    
    raise Exception(f"Loader {dataset_type} not found")
