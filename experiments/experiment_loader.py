from debate import (
    AgentConfig,
    BestOfNDebater,
    BranchedJudge,
    Debater,
    DebateRound,
    HumanDebater,
    Judge,
    MultiRoundBranchingSetting,
    QuestionMetadata,
    SpeechFormatStructure,
)
from data import DatasetConfig, loader_utils, RawDataset, SplitType
from models import Model, ModelSettings, ModelType, ModelUtils, OfflineModelHelper
from prompts import PromptConfig, PromptLoadingConfig, PromptParser
from utils import InputType, input_utils, logger_utils
import utils.constants as constants

from pydantic import BaseModel, model_validator, field_validator
import yaml

from enum import auto, Enum
from typing import Optional
import itertools


class AgentsConfig(BaseModel):
    debaters: list[AgentConfig]
    judge: AgentConfig


class PreviousRunConfig(BaseModel):
    file_path: str | list[str]
    replicate_topics: bool = False
    merge_results: bool = False

    @field_validator("file_path", mode="before")
    @classmethod
    def validate_file_path(cls, file_path: str | list[str]):
        if isinstance(file_path, str):
            return [file_path]
        return file_path


class TournamentType(Enum):
    ROUND_ROBIN = auto()
    SELF_PLAY_ONLY = auto()
    CUSTOM = auto()
    CAPPED_ROUND_ROBIN = auto()
    REPLICATION = auto()


class TournamentConfig(BaseModel):
    tournament_type: TournamentType = TournamentType.ROUND_ROBIN
    custom_matchups: Optional[list[tuple[str, str]]] = None
    replication_file_paths: list[str] = []

    @field_validator("tournament_type", mode="before")
    @classmethod
    def validate_tournament_type(cls, tournament_type: str | TournamentType):
        if isinstance(tournament_type, str):
            tournament_type = TournamentType[tournament_type.upper()]
        return tournament_type

    @model_validator(mode="after")
    @classmethod
    def verify_custom_settings(cls, config):
        if config.custom_matchups and config.tournament_type not in [
            TournamentType.CUSTOM,
            TournamentType.CAPPED_ROUND_ROBIN,
        ]:
            raise ValueError(
                "One cannot set custom matchups if one does not select the custom or capped round robin tournament type"
            )
        elif not config.custom_matchups and config.tournament_type == TournamentType.CUSTOM:
            raise ValueError("One cannot set the custom tournament type without setting custom matchups")
        elif config.replication_file_paths and config.tournament_type != TournamentType.REPLICATION:
            raise ValueError("One cannot use a replication_file_path without using the replication tournament type")
        elif config.tournament_type == TournamentType.REPLICATION and not config.replication_file_paths:
            raise ValueError("One cannot set the replication tournament type without setting replication file paths")
        return config


class ExperimentConfig(BaseModel):
    batch_size: int
    num_speeches: int
    flip: bool = False
    alternate: bool = False
    prompt_config: PromptLoadingConfig = PromptLoadingConfig()
    agents: AgentsConfig
    dataset: DatasetConfig
    annotations_classifier_file_path: Optional[str] = None
    enable_self_debate: bool = False
    previous_run: Optional[PreviousRunConfig] = None
    tournament: Optional[TournamentConfig] = TournamentConfig()
    speech_structure: SpeechFormatStructure = SpeechFormatStructure.DEFAULT_DEBATE
    multi_round_branching: MultiRoundBranchingSetting = MultiRoundBranchingSetting.NONE
    convert_to_double_consultancy: bool = False

    @field_validator("speech_structure", mode="before")
    @classmethod
    def validate_speech_structure(cls, speech_structure: str | SpeechFormatStructure):
        if isinstance(speech_structure, str):
            return SpeechFormatStructure[speech_structure.upper()]
        return speech_structure

    @field_validator("multi_round_branching", mode="before")
    @classmethod
    def validate_multi_round_branching(cls, multi_round_branching: str | MultiRoundBranchingSetting):
        if isinstance(multi_round_branching, str):
            return MultiRoundBranchingSetting[multi_round_branching.upper()]
        return multi_round_branching

    @model_validator(mode="after")
    def check_fields(cls, config):
        if config.flip and config.alternate:
            raise ValueError("flip and alternate cannot both be True at the same time")
        if config.convert_to_double_consultancy and config.speech_structure.num_participants == 1:
            raise ValueError("if convert_to_double_consultancy is used, then a debate format should be used")
        return config


class ExperimentLoader:
    @classmethod
    def merge_debate_rounds(cls, debate_rounds: list[DebateRound]) -> DebateRound:
        """Combines the listed debate rounds into one (batched) debate round"""

        def validate() -> None:
            for debate_round in debate_rounds:
                if (
                    debate_rounds[0].first_debater.name != debate_round.first_debater.name
                    or debate_rounds[0].second_debater.name != debate_round.second_debater.name
                    or debate_rounds[0].judge.name != debate_round.judge.name
                ):
                    raise Exception("Cannot merge rounds of across names")

        validate()
        first_debater_prompts = []
        second_debater_prompts = []
        judge_prompts = []
        metadata_list = []
        first_debater_model = None
        second_debater_model = None
        judge_model = None
        for debate_round in debate_rounds:
            for prompt in debate_round.first_debater.prompts:
                first_debater_prompts.append(prompt)
            for prompt in debate_round.second_debater.prompts:
                second_debater_prompts.append(prompt)
            for prompt in debate_round.judge.prompts:
                judge_prompts.append(prompt)
            for metadata in debate_round.metadata:
                metadata_list.append(metadata)

            first_debater_model = (
                debate_round.first_debater.model
                if not first_debater_model
                else first_debater_model.merge(debate_round.first_debater.model)
            )

            second_debater_model = (
                debate_round.second_debater.model
                if not second_debater_model
                else second_debater_model.merge(debate_round.second_debater.model)
            )

            judge_model = debate_round.judge.model if not judge_model else judge_model.merge(debate_round.judge.model)

        first_debater = debate_rounds[0].first_debater.copy(prompts=first_debater_prompts)
        second_debater = debate_rounds[0].second_debater.copy(prompts=second_debater_prompts)
        judge = debate_rounds[0].judge.copy(prompts=judge_prompts)
        first_debater.model = first_debater_model
        second_debater.model = second_debater_model
        judge.model = judge_model

        return DebateRound(first_debater=first_debater, second_debater=second_debater, judge=judge, metadata=metadata_list)

    @classmethod
    def create_dataset(cls, experiment: ExperimentConfig) -> RawDataset:
        dataset_config = experiment.dataset
        loader_cls = loader_utils.get_loader_type(dataset_config.dataset_type)
        return loader_cls.load(
            full_dataset_filepath=dataset_config.full_dataset_file_path,
            train_filepath=dataset_config.train_file_path,
            val_filepath=dataset_config.val_file_path,
            test_filepath=dataset_config.test_file_path,
            supplemental_file_paths=dataset_config.supplemental_file_paths,
            combine_train_and_val=dataset_config.combine_train_and_val,
            flip_sides=dataset_config.flip_sides,
            shuffle_deterministically=dataset_config.shuffle_deterministically,
        )

    @classmethod
    def get_model_id(cls, model_settings: ModelSettings):
        return f"{model_settings.model_type}_{model_settings.model_file_path}"

    @classmethod
    def create_debate_rounds_for_combination(
        cls,
        experiment: ExperimentConfig,
        dataset: RawDataset,
        split_type: SplitType,
        debater_idxs: tuple[int, int],
        count: int,
        start_idx: int,
        model_cache: Optional[dict[str, Model]] = None,
        offline_model_helper_cache: Optional[dict[str, OfflineModelHelper]] = None,
    ) -> tuple[list[DebateRound], dict[str, Model], dict[str, OfflineModelHelper]]:
        """
        Creates a set of debate round for the specific debaters listed in debater_idxs.

        Params:
            experiment: the configuration for the set of debate rounds
            dataset: the dataset from which one draws the questions and positions
            split_type: whether the quesitons/positions should be sampled from the train, val, or test sets
            debater_idxs: which pair of debaters from the experiment config should we be creating debate rounds for
            count: the number of rounds to create. If <0, it goes through every round in the dataset. This
                is not recommended unless you are replaying rounds offline
            start_idx: the index in the dataset of the first item
            model_cache: a dictionary mapping a model alias (string) to a model. This is useful so that we do not
                instantiate the same model multiple times if this function is called multiple times in a larger
                tournament (it is not needed if you only invoke the function on one pair of models).
            offline_model_helper_cache: similar to the model_cache, but this is for the offline model helper.
                In this case, it maps filepaths to models
        Returns:
            batched_rounds: a list of debate rounds based on the inputted configuration
            model_cache: a cached set of the models used in these debate rounds (useful if you invoke this
                function again).
        """

        # create logger
        logger = logger_utils.get_default_logger(__name__)

        first_alias = experiment.agents.debaters[debater_idxs[0]].model_settings.alias
        second_alias = experiment.agents.debaters[debater_idxs[1]].model_settings.alias

        debater_one_model_id = ExperimentLoader.get_model_id(experiment.agents.debaters[debater_idxs[0]].model_settings)
        debater_two_model_id = ExperimentLoader.get_model_id(experiment.agents.debaters[debater_idxs[1]].model_settings)
        judge_model_id = ExperimentLoader.get_model_id(experiment.agents.judge.model_settings)

        if not model_cache:
            model_cache = {}

        debater_one_model = (
            ModelUtils.instantiate_model(
                model_settings=experiment.agents.debaters[debater_idxs[0]].model_settings,
                is_debater=True,
            )
            if debater_one_model_id not in model_cache
            else model_cache[debater_one_model_id].copy(
                alias=experiment.agents.debaters[debater_idxs[0]].model_settings.alias, is_debater=True
            )
        )
        if debater_one_model:
            model_cache[debater_one_model_id] = debater_one_model

        debater_two_model = (
            ModelUtils.instantiate_model(
                model_settings=experiment.agents.debaters[debater_idxs[1]].model_settings,
                is_debater=True,
            )
            if debater_two_model_id not in model_cache
            else model_cache[debater_two_model_id].copy(
                alias=experiment.agents.debaters[debater_idxs[1]].model_settings.alias,
                is_debater=True,
                nucleus=experiment.agents.debaters[debater_idxs[1]].model_settings.nucleus,
            )
        )
        if debater_two_model:
            model_cache[debater_two_model_id] = debater_two_model

        judge_model = (
            ModelUtils.instantiate_model(
                model_settings=experiment.agents.judge.model_settings,
                is_debater=False,
            )
            if judge_model_id not in model_cache
            else model_cache[judge_model_id].copy(alias=experiment.agents.judge.model_settings.alias, is_debater=False)
        )
        if judge_model:
            model_cache[judge_model_id] = judge_model

        # instantiates offline model helper
        offline_model_helper_cache = offline_model_helper_cache or {}
        offline_model_helpers = None
        first_offline_file_path = experiment.agents.debaters[debater_idxs[0]].model_settings.offline_file_path
        second_offline_file_path = experiment.agents.debaters[debater_idxs[1]].model_settings.offline_file_path
        if (
            first_offline_file_path
            or second_offline_file_path
            or (experiment.previous_run and experiment.previous_run.replicate_topics)
        ):
            file_paths = [first_offline_file_path, second_offline_file_path] + (
                experiment.previous_run.file_path
                if experiment.previous_run
                and (experiment.previous_run.replicate_topics or experiment.previous_run.exclude_topics)
                else []
            )
            offline_model_helpers = []
            for fp in filter(lambda x: x is not None, file_paths):
                if fp in offline_model_helper_cache:
                    offline_model_helpers.append(offline_model_helper_cache[fp])
                else:
                    helper = OfflineModelHelper(file_path_prefix=fp, dataset=dataset, split_type=split_type)
                    offline_model_helpers.append(helper)
                    offline_model_helper_cache[fp] = helper

            for i, helper_one in enumerate(offline_model_helpers):
                for helper_two in offline_model_helpers[i + 1 :]:
                    OfflineModelHelper.reduce_to_common_rounds(helper_one=helper_one, helper_two=helper_two)

        if count < 0:
            if experiment.prompt_config.use_hardcoded_topics:
                count = 1
            elif offline_model_helpers:
                count = offline_model_helpers[0].get_size() * abs(count)
                if experiment.convert_to_double_consultancy:
                    count //= 2
            else:
                count = len(dataset.get_data(experiment.dataset.split_type)) * abs(count)

        logger.info(f"Creating {count} rounds between {first_alias} and {second_alias}")

        # create debate rounds
        rounds = []
        for round_idx in range(count):
            i = round_idx if not offline_model_helpers or not experiment.convert_to_double_consultancy else (round_idx * 2)
            include_first_round = (not experiment.alternate) or i % 2 == 0
            include_flipped_round = experiment.flip or (experiment.alternate and i % 2 == 1)

            if experiment.prompt_config.use_hardcoded_topics:
                topic = experiment.prompt_config.hardcoded_topic_config.topic
                position = experiment.prompt_config.hardcoded_topic_config.positions[0]
                opponent_position = experiment.prompt_config.hardcoded_topic_config.positions[1]
                background_text = constants.DEFAULT_BACKGROUND_TEXT
                title = ""
                correct_index = None
                speeches = []
            else:
                example = (
                    dataset.get_example(idx=i + start_idx, split=split_type)
                    if not offline_model_helpers
                    else offline_model_helpers[0].get_example(idx=i, split_type=split_type)
                )
                topic = example.question
                position = example.positions[0]
                opponent_position = example.positions[1]
                background_text = example.background_text
                title = example.story_title
                correct_index = example.correct_index
                speeches = example.speeches
            debate_identifier = f"{title}_{topic}"

            config_a = PromptConfig(
                name=constants.DEFAULT_DEBATER_A_NAME,
                opponent_name=constants.DEFAULT_DEBATER_B_NAME,
                position=position,
                opponent_position=opponent_position,
                topic=topic,
                background_text=background_text if not experiment.prompt_config.is_memorized else title,
            )

            config_b = PromptConfig(
                name=constants.DEFAULT_DEBATER_B_NAME,
                opponent_name=constants.DEFAULT_DEBATER_A_NAME,
                position=opponent_position,
                opponent_position=position,
                topic=topic,
                background_text=background_text if not experiment.prompt_config.is_memorized else title,
            )

            config_b_first = PromptConfig(
                name=constants.DEFAULT_DEBATER_B_NAME,
                opponent_name=constants.DEFAULT_DEBATER_A_NAME,
                position=opponent_position if experiment.flip else position,
                opponent_position=position if experiment.flip else opponent_position,
                topic=topic,
                background_text=background_text if not experiment.prompt_config.is_memorized else title,
            )

            prompt_a = PromptParser.parse(
                prompt_config=config_a,
                prompts_file_path=experiment.prompt_config.file_path,
                name=experiment.agents.debaters[debater_idxs[0]].model_settings.override_prompt
                or experiment.speech_structure.default_prompt_name
                or experiment.prompt_config.default_prompt_name,
            )

            flipped_prompt_a = PromptParser.parse(
                prompt_config=config_a,
                prompts_file_path=experiment.prompt_config.file_path,
                name=experiment.agents.debaters[debater_idxs[1]].model_settings.override_prompt
                or experiment.speech_structure.default_prompt_name
                or experiment.prompt_config.default_prompt_name,
            )

            prompt_b = PromptParser.parse(
                prompt_config=config_b,
                prompts_file_path=experiment.prompt_config.file_path,
                name=experiment.agents.debaters[debater_idxs[1]].model_settings.override_prompt
                or experiment.speech_structure.default_prompt_name
                or experiment.prompt_config.default_prompt_name,
            )

            flipped_prompt_b = PromptParser.parse(
                prompt_config=config_b,
                prompts_file_path=experiment.prompt_config.file_path,
                name=experiment.agents.debaters[debater_idxs[0]].model_settings.override_prompt
                or experiment.speech_structure.default_prompt_name
                or experiment.prompt_config.default_prompt_name,
            )

            prompt_judge = PromptParser.parse(
                prompt_config=config_a,
                prompts_file_path=experiment.prompt_config.file_path,
                name=experiment.speech_structure.default_prompt_name or experiment.prompt_config.default_prompt_name,
            )

            flipped_prompt_judge = PromptParser.parse(
                prompt_config=config_a if not experiment.speech_structure.flip_position_order else config_b_first,
                prompts_file_path=experiment.prompt_config.file_path,
                name=experiment.speech_structure.default_prompt_name or experiment.prompt_config.default_prompt_name,
            )

            question_metadata = QuestionMetadata(
                first_debater_correct=correct_index == 0,
                question_idx=i,
                background_text=background_text,
                question=topic,
                first_debater_answer=position,
                second_debater_answer=opponent_position,
                debate_identifier=debate_identifier,
            )

            debater_a = Debater(
                name=constants.DEFAULT_DEBATER_A_NAME,
                prompt=prompt_a,
                model=debater_one_model,
                num_speeches=experiment.num_speeches,
                speech_format=experiment.speech_structure.debater_format.get_speech_format(
                    name=constants.DEFAULT_DEBATER_A_NAME,
                    num_speeches=experiment.num_speeches,
                    use_scratchpad=experiment.agents.debaters[debater_idxs[0]].scratchpad.use_scratchpad,
                ),
                scratchpad_config=experiment.agents.debaters[debater_idxs[0]].scratchpad,
                quotes_require_validation=experiment.agents.debaters[
                    debater_idxs[0]
                ].model_settings.require_quote_validation,
            )

            debater_b = Debater(
                name=constants.DEFAULT_DEBATER_B_NAME,
                prompt=prompt_b,
                model=debater_two_model,
                num_speeches=experiment.num_speeches,
                speech_format=experiment.speech_structure.debater_format.get_speech_format(
                    name=constants.DEFAULT_DEBATER_B_NAME,
                    num_speeches=experiment.num_speeches,
                    use_scratchpad=experiment.agents.debaters[debater_idxs[1]].scratchpad.use_scratchpad,
                ),
                scratchpad_config=experiment.agents.debaters[debater_idxs[1]].scratchpad,
                quotes_require_validation=experiment.agents.debaters[
                    debater_idxs[1]
                ].model_settings.require_quote_validation,
            )

            judge = Judge(
                name=constants.DEFAULT_JUDGE_NAME,
                prompt=prompt_judge,
                model=judge_model,
                speech_format=experiment.speech_structure.judge_format.get_speech_format(
                    name=constants.DEFAULT_JUDGE_NAME,
                    num_speeches=experiment.num_speeches,
                    use_scratchpad=experiment.agents.judge.scratchpad.use_scratchpad,
                    flipped=False,
                ),
                num_speeches=experiment.num_speeches,
                scratchpad_config=experiment.agents.judge.scratchpad,
            )

            debate_round = DebateRound(
                first_debater=debater_a,
                second_debater=debater_b,
                judge=judge,
                metadata=[question_metadata],
            )

            flipped_question_metadata = QuestionMetadata(
                first_debater_correct=correct_index == 0,
                question_idx=i,
                background_text=background_text,
                question=topic,
                first_debater_answer=position if not experiment.speech_structure.flip_position_order else opponent_position,
                second_debater_answer=opponent_position if not experiment.speech_structure.flip_position_order else position,
                debate_identifier=debate_identifier,
            )

            flipped_debater_a = Debater(
                name=constants.DEFAULT_DEBATER_A_NAME,
                prompt=flipped_prompt_a,
                model=debater_two_model,
                num_speeches=experiment.num_speeches,
                scratchpad_config=experiment.agents.debaters[debater_idxs[1]].scratchpad,
                speech_format=experiment.speech_structure.debater_format.get_speech_format(
                    name=constants.DEFAULT_DEBATER_A_NAME,
                    num_speeches=experiment.num_speeches,
                    use_scratchpad=experiment.agents.debaters[debater_idxs[1]].scratchpad.use_scratchpad,
                ),
                quotes_require_validation=experiment.agents.debaters[
                    debater_idxs[1]
                ].model_settings.require_quote_validation,
            )

            flipped_debater_b = Debater(
                name=constants.DEFAULT_DEBATER_B_NAME,
                prompt=flipped_prompt_b,
                model=debater_one_model,
                num_speeches=experiment.num_speeches,
                scratchpad_config=experiment.agents.debaters[debater_idxs[0]].scratchpad,
                speech_format=experiment.speech_structure.debater_format.get_speech_format(
                    name=constants.DEFAULT_DEBATER_B_NAME,
                    num_speeches=experiment.num_speeches,
                    use_scratchpad=experiment.agents.debaters[debater_idxs[0]].scratchpad.use_scratchpad,
                ),
                quotes_require_validation=experiment.agents.debaters[
                    debater_idxs[0]
                ].model_settings.require_quote_validation,
            )

            flipped_judge = Judge(
                name=constants.DEFAULT_JUDGE_NAME,
                prompt=flipped_prompt_judge,
                model=judge_model,
                speech_format=experiment.speech_structure.judge_format.get_speech_format(
                    name=constants.DEFAULT_JUDGE_NAME,
                    num_speeches=experiment.num_speeches,
                    use_scratchpad=experiment.agents.judge.scratchpad.use_scratchpad,
                    flipped=True,
                ),
                num_speeches=experiment.num_speeches,
                scratchpad_config=experiment.agents.judge.scratchpad,
            )

            flipped_round = DebateRound(
                first_debater=flipped_debater_a,
                second_debater=flipped_debater_b,
                judge=flipped_judge,
                metadata=[flipped_question_metadata],
            )

            if first_offline_file_path:
                helper = next((x for x in offline_model_helpers if x.file_path_prefix == first_offline_file_path))
                if include_first_round:
                    debate_round.first_debater.model = helper.create_offline_model(
                        alias=experiment.agents.debaters[debater_idxs[0]].model_settings.alias,
                        debater_name=debate_round.first_debater.name,
                        idx=i,
                        positions=(position, opponent_position),
                        best_of_n_config=experiment.agents.debaters[debater_idxs[0]].best_of_n,
                    )
                if include_flipped_round:
                    flipped_round.second_debater.model = helper.create_offline_model(
                        alias=experiment.agents.debaters[debater_idxs[0]].model_settings.alias,
                        debater_name=debate_round.second_debater.name,
                        idx=i,
                        positions=(position, opponent_position)
                        if not experiment.speech_structure.flip_position_order
                        else (opponent_position, position),
                        best_of_n_config=experiment.agents.debaters[debater_idxs[0]].best_of_n,
                    )
                if (
                    experiment.speech_structure.flip_position_order and not include_first_round
                ):  # if the first debater speeches are missing in consultancy, then we should expect B to go first
                    for metadata in flipped_round.metadata:
                        metadata.first_debater_correct = not metadata.first_debater_correct

            if second_offline_file_path:
                helper = next((x for x in offline_model_helpers if x.file_path_prefix == second_offline_file_path))

                if include_first_round and experiment.speech_structure.num_participants > 1:
                    debate_round.second_debater.model = helper.create_offline_model(
                        alias=experiment.agents.debaters[debater_idxs[1]].model_settings.alias,
                        debater_name=flipped_round.second_debater.name,
                        idx=i if not experiment.convert_to_double_consultancy else (i + 1),
                        positions=(position, opponent_position),
                        best_of_n_config=experiment.agents.debaters[debater_idxs[1]].best_of_n,
                    )
                if include_flipped_round and experiment.speech_structure.num_participants > 1:
                    flipped_round.first_debater.model = helper.create_offline_model(
                        alias=experiment.agents.debaters[debater_idxs[1]].model_settings.alias,
                        debater_name=flipped_round.first_debater.name,
                        idx=i if not experiment.convert_to_double_consultancy else (i + 1),
                        positions=(position, opponent_position)
                        if not experiment.speech_structure.flip_position_order
                        else (opponent_position, position),
                        best_of_n_config=experiment.agents.debaters[debater_idxs[1]].best_of_n,
                    )
                if (
                    experiment.speech_structure.flip_position_order
                    and not include_first_round
                    and not first_offline_file_path
                ):
                    for metadata in flipped_round.metadata:
                        metadata.first_debater_correct = not metadata.first_debater_correct

            original_first_debater = debate_round.first_debater
            original_second_debater = debate_round.second_debater
            original_flipped_first_debater = flipped_round.first_debater
            original_flipped_second_debater = flipped_round.second_debater

            if experiment.agents.debaters[debater_idxs[0]].best_of_n and (
                not first_offline_file_path or experiment.agents.debaters[debater_idxs[0]].best_of_n.recompute
            ):
                debate_round.set_first_debater(
                    BestOfNDebater(
                        debater=debate_round.first_debater,
                        opposing_debater=original_second_debater,
                        judge=debate_round.judge,
                        best_of_n_config=experiment.agents.debaters[debater_idxs[0]].best_of_n,
                        background_text=question_metadata.background_text,
                    )
                )
                if experiment.flip or experiment.alternate:
                    flipped_round.set_second_debater(
                        BestOfNDebater(
                            debater=flipped_round.second_debater,
                            opposing_debater=original_flipped_first_debater,
                            judge=debate_round.judge,
                            best_of_n_config=experiment.agents.debaters[debater_idxs[0]].best_of_n,
                            background_text=question_metadata.background_text,
                        )
                    )
            if experiment.agents.debaters[debater_idxs[1]].best_of_n and (
                not second_offline_file_path or experiment.agents.debaters[debater_idxs[1]].best_of_n.recompute
            ):
                if experiment.speech_structure.num_participants > 1:
                    debate_round.set_second_debater(
                        BestOfNDebater(
                            debater=debate_round.second_debater,
                            opposing_debater=original_first_debater,
                            judge=debate_round.judge,
                            best_of_n_config=experiment.agents.debaters[debater_idxs[1]].best_of_n,
                            background_text=question_metadata.background_text,
                        )
                    )
                if experiment.flip or experiment.alternate:
                    flipped_round.set_first_debater(
                        BestOfNDebater(
                            debater=flipped_round.first_debater,
                            opposing_debater=original_flipped_second_debater,
                            judge=debate_round.judge,
                            best_of_n_config=experiment.agents.debaters[debater_idxs[1]].best_of_n,
                            background_text=question_metadata.background_text,
                        )
                    )

            if experiment.agents.debaters[debater_idxs[0]].model_settings.is_human:
                debate_round.set_first_debater(HumanDebater(debater=debate_round.first_debater, speeches=speeches))
                flipped_round.set_second_debater(HumanDebater(debater=flipped_round.second_debater, speeches=speeches))
            if experiment.agents.debaters[debater_idxs[1]].model_settings.is_human:
                debate_round.set_second_debater(HumanDebater(debater=debate_round.second_debater, speeches=speeches))
                flipped_round.set_first_debater(HumanDebater(debater=flipped_round.first_debater, speeches=speeches))

            if experiment.multi_round_branching != MultiRoundBranchingSetting.NONE:
                assert experiment.batch_size == 1, "Multi round branching only supports a batch size of 1"
                debate_round.set_judge(
                    BranchedJudge(
                        judge=debate_round.judge,
                        debater_one=debate_round.first_debater,
                        debater_two=debate_round.second_debater,
                        setting=experiment.multi_round_branching,
                        speeches_per_round=experiment.speech_structure.num_participants,
                        flip_first_debater=True,
                    )
                )
                flipped_round.set_judge(
                    BranchedJudge(
                        judge=flipped_round.judge,
                        debater_one=flipped_round.first_debater,
                        debater_two=flipped_round.second_debater,
                        setting=experiment.multi_round_branching,
                        speeches_per_round=experiment.speech_structure.num_participants,
                        flip_first_debater=False,
                    )
                )

            if include_first_round:
                rounds.append(debate_round)
            if include_flipped_round:
                rounds.append(flipped_round)

        if len(rounds) <= 1:
            return rounds, model_cache, offline_model_helper_cache

        if experiment.batch_size == 1:
            return rounds, model_cache, offline_model_helper_cache

        # batches the debate rounds for efficient generation
        batched_rounds = []
        current_normal_batch = []
        current_flipped_batch = []
        for i, debate_round in enumerate(rounds):
            if i % 2 == 0 or (not experiment.flip and not experiment.alternate):
                current_normal_batch.append(debate_round)
                if len(current_normal_batch) == experiment.batch_size:
                    batched_rounds.append(ExperimentLoader.merge_debate_rounds(current_normal_batch))
                    current_normal_batch = []
            else:
                current_flipped_batch.append(debate_round)
                if len(current_flipped_batch) == experiment.batch_size:
                    batched_rounds.append(ExperimentLoader.merge_debate_rounds(current_flipped_batch))
                    current_flipped_batch = []

        if current_normal_batch:
            batched_rounds.append(ExperimentLoader.merge_debate_rounds(current_normal_batch))
        if current_flipped_batch:
            batched_rounds.append(ExperimentLoader.merge_debate_rounds(current_flipped_batch))

        return batched_rounds, model_cache, offline_model_helper_cache

    @classmethod
    def get_debater_combinations(
        cls, experiment: ExperimentConfig, count: int = -1, dataset: Optional[RawDataset] = None
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """Returns all the combinations of debaters that would need to debate each other in a round robin tournament"""
        logger = logger_utils.get_default_logger(__name__)

        if not experiment.agents:
            raise Exception("A judge must be defined")
        if (
            not experiment.agents.debaters or len(experiment.agents.debaters) < 1
        ) and experiment.tournament.tournament_type != TournamentType.REPLICATION:
            raise Exception("At least 1 debater must be defined")

        default_start_idxs = [(0, count) for i in range(len(experiment.agents.debaters))]
        if (
            experiment.tournament.tournament_type == TournamentType.SELF_PLAY_ONLY
            or experiment.speech_structure.num_participants == 1
        ):
            return zip([(i, i) for i in range(len(experiment.agents.debaters))], default_start_idxs)
        elif experiment.tournament.tournament_type == TournamentType.ROUND_ROBIN:
            all_idxs = [i for i in range(len(experiment.agents.debaters))] if len(experiment.agents.debaters) > 1 else [0, 0]
            all_debater_idxs = [elem for elem in itertools.combinations(all_idxs, r=2)]
            if experiment.enable_self_debate and len(experiment.agents.debaters) > 1:
                all_debater_idxs += [(idx, idx) for idx in all_idxs]
            return zip(all_debater_idxs, default_start_idxs)
        elif experiment.tournament.tournament_type == TournamentType.CAPPED_ROUND_ROBIN:
            all_idxs = [i for i in range(len(experiment.agents.debaters))] if len(experiment.agents.debaters) > 1 else [0, 0]
            all_debater_idxs = [elem for elem in itertools.combinations(all_idxs, r=2)]
            if experiment.enable_self_debate and len(experiment.agents.debaters) > 1:
                all_debater_idxs += [(idx, idx) for idx in all_idxs]

            dataset_length = len(dataset.get_data(split=experiment.dataset.split_type))
            section_length = dataset_length // len(experiment.agents.debaters)
            start_idxs = []
            current = 0
            length = len(all_idxs) - 1
            for row in range(len(all_idxs) - 1):
                for i in range(len(all_idxs) - row - 1):
                    section = (current + i) % len(all_idxs)
                    start_idxs.append((section * section_length, section_length))
                current = (current + 2) % len(all_idxs)

            if experiment.tournament.custom_matchups:
                matchup_to_key_fn = lambda x, y: "_v_".join(sorted([str(x), str(y)]))
                matchup_set = set([matchup_to_key_fn(a, b) for a, b in experiment.tournament.custom_matchups])
                new_debater_idxs = []
                new_start_idxs = []
                for debater_idxs, start_idx in zip(all_debater_idxs, start_idxs):
                    if (
                        matchup_to_key_fn(
                            experiment.agents.debaters[debater_idxs[0]].model_settings.alias,
                            experiment.agents.debaters[debater_idxs[1]].model_settings.alias,
                        )
                        in matchup_set
                    ):
                        new_debater_idxs.append(debater_idxs)
                        new_start_idxs.append(start_idx)
                return zip(new_debater_idxs, new_start_idxs)
            return zip(all_debater_idxs, start_idxs)
        elif experiment.tournament.tournament_type == TournamentType.CUSTOM:
            matchup_idxs = []
            aliases_to_idxs = {debater.model_settings.alias: i for i, debater in enumerate(experiment.agents.debaters)}
            for a, b in experiment.tournament.custom_matchups:
                if a not in aliases_to_idxs:
                    raise Exception(f"Custom matchup for ({a} v {b}) could not be created because ({a}) was not recognized")
                if b not in aliases_to_idxs:
                    raise Exception(f"Custom matchup for ({a} v {b}) could not be created because ({b}) was not recognized")
                matchup_idxs.append((aliases_to_idxs[a], aliases_to_idxs[b]))
            return zip(matchup_idxs, default_start_idxs)
        elif experiment.tournament.tournament_type == TournamentType.REPLICATION:
            logger.warn('Using a "replication" tournament type will mutate the underlying experiment')
            experiment.flip = False
            experiment.alternate = False
            matchup_idxs = []
            for file_path_root in experiment.tournament.replication_file_paths:
                files = input_utils.read_file_texts(
                    base_path=file_path_root,
                    input_type=InputType.JSON_TRANSCRIPT,
                    include_full_file_path=True,
                )
                run = input_utils.read_file_texts(base_path=file_path_root, input_type=InputType.RUN, should_load=True)[0]
                assert len(run) == len(files), f"Run length ({len(run)}) does not match file length ({len(files)})"
                for (_, row), (_, path) in zip(run.iterrows(), files):
                    first_alias = row["first_debater_alias"]
                    second_alias = row["second_debater_alias"]
                    experiment.agents.debaters.append(
                        AgentConfig(
                            model_settings=ModelSettings(
                                model_type=ModelType.OFFLINE,
                                offline_file_path=path,
                                alias=first_alias,
                                require_quote_validation=False,
                            )
                        )
                    )
                    experiment.agents.debaters.append(
                        AgentConfig(
                            model_settings=ModelSettings(
                                model_type=ModelType.OFFLINE,
                                offline_file_path=path,
                                alias=second_alias,
                                require_quote_validation=False,
                            )
                        )
                    )
                    matchup_idxs.append(((len(experiment.agents.debaters) - 2, len(experiment.agents.debaters) - 1), (0, 1)))

            return matchup_idxs
        else:
            raise Exception("Tournament type was not recognized")

    @classmethod
    def generate_debate_rounds(
        cls, experiment_file_path: str, name: str, count: int = 1
    ) -> tuple[list[DebateRound], ExperimentConfig]:
        """
        Generates a list of debate rounds with the given configuration

        Params:
            experiment_file_path: path to the file of the experiment config
            name: the name of the specific config within the broader config file
            count: the number of debate rounds to create

        Returns:
            all_rounds: a list of (batched) debate rounds constructed using the config
            experiment: the configuration used to create the debate rounds
        """
        # create experiment config
        with open(experiment_file_path) as f:
            loaded_yaml = yaml.safe_load(f)
        name = name or [key for key in loaded_yaml][0]
        experiment = ExperimentConfig(**loaded_yaml[name])

        # create dataset
        dataset = ExperimentLoader.create_dataset(experiment)
        start_idx = 0

        all_rounds = []
        model_cache = {}
        offline_model_helper_cache = {}
        for combination, (start_idx, count_to_use) in ExperimentLoader.get_debater_combinations(
            experiment=experiment, count=count, dataset=dataset
        ):
            rounds, model_cache, offline_model_helper_cache = ExperimentLoader.create_debate_rounds_for_combination(
                experiment=experiment,
                dataset=dataset,
                split_type=experiment.dataset.split_type,
                debater_idxs=combination,
                count=count_to_use,
                start_idx=start_idx,
                model_cache=model_cache,
                offline_model_helper_cache=offline_model_helper_cache,
            )
            all_rounds.extend(rounds)

        return all_rounds, experiment
