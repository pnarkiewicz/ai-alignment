from __future__ import annotations

import copy
import json
import random
from enum import Enum, auto
from typing import Any, Optional

import utils.constants as constants
from debate.agent import Agent, ScratchpadConfig
from debate.speech_format import Speech, SpeechFormat, SpeechFormatType
from debate.transcript import SpeechFormat, Transcript
from models import Model, ModelResponse, SpeechStructure
from prompts import Prompt
from utils import logger_utils


class Judge(Agent):
    def __init__(
        self,
        name: str,
        prompt: Prompt | list[Prompt],
        model: Model,
        num_speeches: int,
        speech_format: Optional[SpeechFormat] = None,
        speech_structure: SpeechStructure = SpeechStructure.DECISION,
        expected_saver: str = constants.DEFAULT_JUDGE_NAME,
        scratchpad_config: ScratchpadConfig = ScratchpadConfig(),
    ):
        """
        The abstraction used to both judge rounds and determine who speaks next.

        Params:
            name: The name of the judge (just needs to be unique within the round)
            prompt: The prompt structure used to generate inputs to the model
            model: The model that actually generates text
            num_speeches: The number of speeches each debater is expected to deliver
            speech_format: the order of speeches the judge expects to hear
            speech_structure: the default way the judge is to supposed to generate text
            expected_saver: whether the judge or the debater is in charge of saving the transcript
            scratchpad_config: configuration that specifies if and how to use a scratchpad
        """
        super().__init__(
            name=name,
            is_debater=False,
            prompt=prompt,
            model=model,
            num_speeches=num_speeches,
            quotes_require_validation=False,
            receive_validated_quotes=True,
            speech_format=speech_format
            if speech_format
            else SpeechFormatType.DEFAULT_DEBATE_JUDGE.get_speech_format(
                name=name,
                num_speeches=num_speeches,
                use_scratchpad=scratchpad_config.use_scratchpad,
            ),
        )
        self.logger = logger_utils.get_default_logger(__name__)
        self.speech_structure = speech_structure
        self.expected_saver = expected_saver
        self.scratchpad_config = scratchpad_config
        self.num_speeches = num_speeches

    def generate(
        self,
        max_new_tokens: Optional[int] = None,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
    ) -> [list[ModelResponse]]:
        """Calls the underlying model to generate text"""
        model_inputs = [transcript.to_model_input() for transcript in self.transcripts]
        max_new_tokens = max_new_tokens if speech_structure == SpeechStructure.OPEN_ENDED else 15
        return self.model.predict(
            inputs=model_inputs,
            max_new_tokens=max_new_tokens or self.speech_format.tokens_per_speech,
            speech_structure=speech_structure,
        )

    def __call__(self) -> list[str | bool]:
        """
        Calls the underlying model to generate text for each element in the batch.

        Returns:
            Either a string with the text it generated or a boolean indicating whether the first debater won
            (depending on whether the speech was a decision or open-ended) for each element in the batch.
        """
        if self.scratchpad_config.use_scratchpad or not self.transcripts[0].only_decision_remains():
            batch_generation = self.generate(max_new_tokens=250, speech_structure=SpeechStructure.OPEN_ENDED)
            batch_reasoning = [response.speech for response in batch_generation]
        if self.transcripts[0].only_decision_remains():  # all formats should be the same so we can use any transcript
            if self.scratchpad_config.use_scratchpad:
                for i, reasoning in enumerate(batch_reasoning):
                    super().receive_message(speaker=self.name, content=reasoning, idx=i)
            batch_predictions = self.generate(max_new_tokens=15, speech_structure=self.speech_structure)
            validated_predictions = self.validate_responses(batch_predictions)
            returned_response = self.process_responses(validated_predictions)
            if self.scratchpad_config.use_scratchpad:
                for generation, prediction in zip(batch_generation, batch_predictions):
                    prediction.failed = generation.failed or prediction.failed

            return returned_response, batch_predictions
        return batch_reasoning, batch_generation

    def validate_responses(self, responses: list[ModelResponse]) -> None:
        """Confirms that the responses matched the expected format"""
        for response in filter(lambda x: not x.decision, responses):
            self.logger.warn(f'Response of "{response}" was invalid. Must be a debater name.')
        return responses

    def process_responses(self, responses: list[ModelResponse]) -> list[Any]:
        """Converts a text response to a list of booleans indicating if Debater_A won"""
        return [constants.DEFAULT_DEBATER_A_NAME in response.decision for response in responses]

    def copy(
        self,
        transcripts: Optional[list[Transcript]] = None,
        prompts: Optional[list[Prompt] | Prompt] = None,
    ) -> Judge:
        """Deep copies everything except the underlying model"""
        judge = Judge(
            name=self.name,
            prompt=prompts if prompts else [copy.deepcopy(prompt) for prompt in self.prompts],
            model=self.model,
            num_speeches=self.num_speeches,
            speech_format=self.speech_format,
            speech_structure=self.speech_structure,
            expected_saver=self.expected_saver,
            scratchpad_config=self.scratchpad_config,
        )
        if transcripts:
            judge.transcripts = [transcript.copy() for transcript in transcripts]
        return judge


class MultiRoundBranchingSetting(Enum):
    NONE = auto()
    FULL = auto()
    HALF = auto()
    SINGLE_RANDOM = auto()


class BranchedJudge(Judge):
    NUM_BRANCHES = 2  # making this a class variable b/c it's not configurable at the moment

    def __init__(
        self,
        judge: Judge,
        debater_one: Debater,
        debater_two: Debater,
        setting: MultiRoundBranchingSetting = MultiRoundBranchingSetting.FULL,
        speeches_per_round: int = 2,
        flip_first_debater: bool = True,
    ):
        self.setting = setting
        self.debater_one = debater_one
        self.debater_two = debater_two
        self.internal_judge = judge
        self.empty_debater_one_transcript = copy.deepcopy(debater_one.transcripts[0])
        self.empty_debater_two_transcript = copy.deepcopy(debater_two.transcripts[0])
        self.empty_judge_transcript = copy.deepcopy(judge.transcripts[0])
        self.flip_first_debater = flip_first_debater

        self.speeches_per_round = speeches_per_round
        self.num_rounds = self.internal_judge.speech_format.num_speeches
        self.num_transcripts = BranchedJudge.NUM_BRANCHES ** (
            max(1, 2 * (self.num_rounds - (1 if self.speeches_per_round == 1 else 0)))
        )
        self.transcript_idx_to_speech_idx = {
            i: self.__get_speeches_for_transcript(i) for i in range(self.num_transcripts)
        }
        self.max_expected_speeches = max([max(val) + 1 for val in self.transcript_idx_to_speech_idx.values()])
        self.expected_transcripts = self.__get_expected_transcripts()

        self.received_speeches = [None for i in range(self.max_expected_speeches)]
        self.completed_transcripts = []

        super().__init__(
            name=judge.name,
            prompt=judge.prompts,
            model=judge.model,
            num_speeches=judge.num_speeches,
            speech_format=judge.speech_format,
            speech_structure=judge.speech_structure,
            expected_saver=judge.expected_saver,
            scratchpad_config=judge.scratchpad_config,
        )

    def copy(
        self,
        transcripts: Optional[list[Transcript]] = None,
        prompts: Optional[list[Prompt] | Prompt] = None,
    ) -> Judge:
        """Deep copies everything except the underlying model"""
        return BranchedJudge(
            judge=self.internal_judge.copy(transcripts=transcripts, prompts=prompts),
            debater_one=self.debater_one.copy(),
            debater_two=self.debater_two.copy(),
            setting=self.setting,
            speeches_per_round=self.speeches_per_round,
        )

    def post_speech_processing(self):
        if not self.internal_judge.get_next_expected_speaker():
            self.completed_transcripts.append(copy.deepcopy(self.internal_judge.transcripts[0]))
            next_idx = self.expected_transcripts[len(self.completed_transcripts) % len(self.expected_transcripts)]
            self.__reset_agent_transcript(
                agent=self.debater_one,
                blank_transcript=self.empty_debater_one_transcript,
                idx=next_idx,
            )
            self.__reset_agent_transcript(
                agent=self.debater_two,
                blank_transcript=self.empty_debater_two_transcript,
                idx=next_idx,
            )
            self.__reset_agent_transcript(
                agent=self.internal_judge,
                blank_transcript=self.empty_judge_transcript,
                idx=next_idx,
            )

    def get_next_expected_speaker(self):
        if len(self.completed_transcripts) >= len(self.expected_transcripts):
            return None
        return self.internal_judge.get_next_expected_speaker()

    def receive_message(
        self,
        speaker: str,
        content: str,
        idx: int,
        supplemental: Optional[dict[Any, Any] | list[dict[Any, Any]]] = None,
    ):
        # ok we have to set where we think the expected speech should be
        if speaker != self.name:
            current_transcript_idx = self.expected_transcripts[len(self.completed_transcripts)]
            current_transcript_length = self.internal_judge.transcripts[0].get_external_speech_count()
            transcript_speech_idxs = self.transcript_idx_to_speech_idx[current_transcript_idx]
            insertion_speech_idx = transcript_speech_idxs[current_transcript_length]
            if not self.received_speeches[insertion_speech_idx]:
                self.received_speeches[insertion_speech_idx] = Speech(
                    speaker=speaker, content=content, supplemental=supplemental
                )
        self.internal_judge.receive_message(speaker=speaker, content=content, idx=idx, supplemental=supplemental)

    def get_transcript(self, idx: int = 0) -> Transcript:
        """Returns the transcript at the specified index"""
        return self.completed_transcripts[idx]

    @property
    def transcripts(self):
        return self.internal_judge.transcripts

    @transcripts.setter
    def transcripts(self, transcripts):
        self.internal_judge.transcripts = transcripts

    def save(self, save_file_path_prefix: str, metadata: Optional[list[dict[Any, Any]]] = None):
        speech_to_scores = {}
        for i, transcript_idx in enumerate(self.expected_transcripts):
            supplemental = self.completed_transcripts[i].get_last_internal_speech().supplemental
            winning_probability = (
                supplemental.probabilistic_decision[constants.DEFAULT_DEBATER_A_NAME]
                if supplemental.probabilistic_decision
                else (1.0 if supplemental.decision == constants.DEFAULT_DEBATER_A_NAME else 0.0)
            )
            speech_idxs = self.transcript_idx_to_speech_idx[transcript_idx]
            for speech_idx in speech_idxs:
                if speech_idx not in speech_to_scores:
                    speech_to_scores[speech_idx] = []
                speech_to_scores[speech_idx].append(winning_probability)

        pairs = []
        for first_idx in filter(lambda x: x % 2 == 0, sorted(list(speech_to_scores.keys()))):
            second_idx = first_idx + 1
            if first_idx in speech_to_scores and second_idx in speech_to_scores:
                first_score = sum(speech_to_scores[first_idx]) / len(speech_to_scores[first_idx])
                second_score = sum(speech_to_scores[second_idx]) / len(speech_to_scores[second_idx])

                speaker = self.received_speeches[first_idx].speaker
                first_score = first_score if speaker == constants.DEFAULT_DEBATER_A_NAME else 1 - first_score
                second_score = second_score if speaker == constants.DEFAULT_DEBATER_A_NAME else 1 - second_score

                preferred_entry = (
                    self.received_speeches[first_idx]
                    if first_score > second_score
                    else self.received_speeches[second_idx]
                )
                rejected_entry = (
                    self.received_speeches[second_idx]
                    if first_score > second_score
                    else self.received_speeches[first_idx]
                )

                new_preferred_entry = Speech(
                    speaker=preferred_entry.speaker,
                    content=preferred_entry.content,
                    supplemental=ModelResponse(
                        speech=preferred_entry.content,
                        preference=max(first_score, second_score),
                        prompt=preferred_entry.supplemental.prompt,
                        rejected_responses=[
                            ModelResponse(
                                speech=rejected_entry.content,
                                preference=min(first_score, second_score),
                                prompt=rejected_entry.supplemental.prompt,
                            )
                        ],
                    ),
                )
                pairs.append(new_preferred_entry.dict())
                assert rejected_entry.supplemental.prompt == preferred_entry.supplemental.prompt, (
                    f"{preferred_entry.supplemental.prompt}\n\n\n===\n\n\n{rejected_entry.supplemental.prompt}"
                )

        full_paired_transcript = {"metadata": metadata, "speeches": pairs}
        with open(save_file_path_prefix + ".json", "w") as f:
            json.dump(full_paired_transcript, f)

    def __get_round_start_idx(self, round_idx: int):
        start_idx = 0
        current_idx = 0
        while round_idx > current_idx:
            current_idx += 1
            start_idx += BranchedJudge.NUM_BRANCHES ** (2 * current_idx)
        return start_idx

    def __get_speeches_for_transcript(self, transcript_idx: int):
        if self.speeches_per_round == 2:
            if self.internal_judge.speech_format.num_speeches == 2:
                first = transcript_idx // 8
                second = 2 if transcript_idx % 8 < 4 else 3
                third = transcript_idx // 4
                segment_start = ((transcript_idx // 4) * 4) + 4
                third = segment_start if transcript_idx % 4 < 2 else segment_start + 1
                fourth = segment_start + (2 if transcript_idx % 2 == 0 else 3)
                return [first, second, third, fourth]
            else:
                first = transcript_idx // 2
                second = 2 + (transcript_idx % 2)
                return [first, second]
        elif self.speeches_per_round == 1:
            if self.internal_judge.speech_format.num_speeches == 2:
                first = transcript_idx // 2
                second = (2 + (first * 2)) + (1 if transcript_idx % 2 == 1 else 0)
            else:
                return [(2 * (transcript_idx // 2)) + (transcript_idx % 2)]
        else:
            raise Exception("Unprocessable number of speeches per round")

    def __reset_agent_transcript(self, agent: Agent, blank_transcript: Transcript, idx: int):
        transcript_to_add = copy.deepcopy(blank_transcript)
        agent.transcripts = [transcript_to_add]
        for speech_idx in self.transcript_idx_to_speech_idx[idx]:
            if self.received_speeches[speech_idx]:
                speech = self.received_speeches[speech_idx]
                agent.receive_message(
                    speaker=speech.speaker,
                    content=speech.content,
                    idx=0,
                    supplemental=speech.supplemental,
                )

    def __get_expected_transcripts(self):
        if self.setting == MultiRoundBranchingSetting.FULL:
            return [i for i in range(self.num_transcripts)]
        elif self.setting == MultiRoundBranchingSetting.HALF:
            if self.flip_first_debater:
                candidates = [0, 2, self.num_transcripts // 2, (self.num_transcripts // 2) + 2]
            else:
                candidates = [0, 1, self.num_transcripts // 4, (self.num_transcripts // 4) + 1]
            return candidates[0:2] if self.internal_judge.speech_format.num_speeches == 1 else candidates
        elif self.setting == MultiRoundBranchingSetting.SINGLE_RANDOM:
            random_number = random.random()
            if random_number < 0.25:
                return [0, self.num_transcripts // 2]
            elif random_number < 0.5:
                return [0, self.num_transcripts // 4]
            elif random_number < 0.75:
                return [0, self.num_transcripts // 8]
            else:
                return [0, self.num_transcripts // 16]
        else:
            raise Exception(f"Multi round branch setting of {self.setting} was not recognized")
