from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel

from debate.debater import Debater
from debate.judge import Judge
from models import ModelResponse
from utils import logger_utils, quote_utils
import utils.constants as constants


class QuestionMetadata(BaseModel):
    first_debater_correct: bool
    question_idx: int
    background_text: str
    question: str
    first_debater_answer: str
    second_debater_answer: str
    debate_identifier: str


class DebateRoundSummary(BaseModel):
    metadata: QuestionMetadata
    transcript: Any
    winning_alias: str
    losing_alias: str
    first_debater_alias: str
    second_debater_alias: str
    first_debater_wins: bool
    judge_alias: str
    winning_debater_prob: float = 1.0
    first_debater_win_prob: float = 0.5
    second_debater_win_prob: float = 0.5
    first_debater_speaks: bool = True
    second_debater_speaks: bool = True
    failed: bool = False


class SplittingRule(Enum):
    OPENING_ONLY = 1
    ALL_RANDOM = 2


class DebateRound:
    def __init__(
        self,
        first_debater: Debater,
        second_debater: Debater,
        judge: Judge,
        metadata: QuestionMetadata | list[QuestionMetadata],
    ):
        """An abstraction that coordinates the ordered generation of speeches by the debaters and the judge."""
        self.first_debater = first_debater
        self.second_debater = second_debater
        self.judge = judge
        self.metadata = metadata if type(metadata) == list else [metadata]
        self.name_to_agent = {
            self.first_debater.name: self.first_debater,
            self.second_debater.name: self.second_debater,
            self.judge.name: self.judge,
        }
        self.logger = logger_utils.get_default_logger(__name__)

    def set_first_debater(self, debater: Debater):
        """Changes the identity of the first debater in the debate."""
        self.first_debater = debater
        self.name_to_agent[self.first_debater.name] = debater

    def set_second_debater(self, debater: Debater):
        """Changes the identity of the second debater in the debate."""
        self.second_debater = debater
        self.name_to_agent[self.second_debater.name] = debater

    def set_judge(self, judge: Judge):
        """Changes the identity of the judge in the debate."""
        self.judge = judge
        self.name_to_agent[self.judge.name] = judge

    def run_round(self) -> tuple[list[str], ModelResponse]:
        """
        Each debater generates speeches until the judge renders their decision.

        Returns:
            last_output: a list of strings with the name of the agent that won each debate in the batch
            last_model_output: the model generation from the judge's decision. This is useful if the judge
                also returns the probability that a given debater won.
        """
        last_output = None
        last_model_output = None
        next_speaker = self.judge.get_next_expected_speaker()
        while next_speaker:
            speaker = self.name_to_agent[next_speaker]
            try:
                batch_response, model_output = speaker()
            except Exception as e:
                self.logger.error("Received an error while trying to generate a speech %s", str(e), exc_info=True)
                return None, None

            for idx, (response, output) in enumerate(zip(batch_response, model_output)):
                validated_response = str(response)
                if speaker.quotes_require_validation:
                    validated_response = quote_utils.validate_and_replace_quotes(
                        speech_content=str(response),
                        background_text=self.metadata[min(idx, len(self.metadata) - 1)].background_text,
                    )
                for _, agent in self.name_to_agent.items():
                    response_to_use = validated_response if agent.receive_validated_quotes else response
                    agent.receive_message(speaker=speaker.name, content=response_to_use, idx=idx, supplemental=output)

            self.judge.post_speech_processing()
            next_speaker = self.judge.get_next_expected_speaker()

            last_output = batch_response
            last_model_output = model_output

        return last_output, last_model_output

    def record_winners(
        self,
        last_output: Optional[list[str]],
        last_model_output: Optional[list[ModelResponse]],
        save_file_path_prefix: Optional[str] = None,
    ) -> list[DebateRoundSummary]:
        """Generates a full summary of the debate round including the winner, transcript, metadata, and aliases of all the participating models"""
        if not last_output:
            return []

        first_debater_win_list = []
        winning_probability_list = []
        failed_list = []
        for i, (debater_a_wins, model_output) in enumerate(zip(last_output, last_model_output)):
            winner = constants.DEFAULT_DEBATER_A_NAME if debater_a_wins else constants.DEFAULT_DEBATER_B_NAME
            first_debater_win_list.append(winner == self.first_debater.name)
            string_value = self.judge.get_transcript(idx=i).full_string_value()
            winning_probability_list.append(1.0 if not model_output.probabilistic_decision else model_output.probabilistic_decision[winner])
            failed_list.append(model_output.failed)
            self.logger.debug(string_value)

        if save_file_path_prefix:
            self.name_to_agent[self.judge.expected_saver].save(save_file_path_prefix=save_file_path_prefix, metadata=[item.dict() for item in self.metadata])

        return [
            DebateRoundSummary(
                metadata=self.metadata[i % len(self.metadata)],
                transcript=self.judge.get_transcript(idx=i),
                winning_alias=self.first_debater.get_alias() if first_debater_wins else self.second_debater.get_alias(),
                losing_alias=self.first_debater.get_alias() if not first_debater_wins else self.second_debater.get_alias(),
                first_debater_alias=self.first_debater.get_alias(),
                second_debater_alias=self.second_debater.get_alias(),
                first_debater_wins=first_debater_wins,
                judge_alias=self.judge.get_alias(),
                winning_debater_prob=winning_probability_list[i],
                first_debater_win_prob=winning_probability_list[i] if first_debater_wins else (1 - winning_probability_list[i]),
                second_debater_win_prob=(1 - winning_probability_list[i]) if first_debater_wins else winning_probability_list[i],
                first_debater_speaks=constants.DEFAULT_DEBATER_A_NAME in self.judge.get_transcript(idx=i).get_speakers(),
                second_debater_speaks=constants.DEFAULT_DEBATER_B_NAME in self.judge.get_transcript(idx=i).get_speakers(),
                failed=failed_list[i],
            )
            for i, first_debater_wins in enumerate(first_debater_win_list)
        ]

    def __call__(self, save_file_path_prefix: Optional[str] = None) -> list[DebateRoundSummary]:
        """Runs the round and generates a summary of the results"""
        last_output, last_model_output = self.run_round()
        return self.record_winners(last_output=last_output, last_model_output=last_model_output, save_file_path_prefix=save_file_path_prefix)
