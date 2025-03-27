from __future__ import annotations

from models import ModelResponse
from prompts import Prompt, PromptTag, RoleType
import utils.constants as constants

from pydantic import BaseModel

from abc import ABC
from enum import Enum, auto
from typing import Callable, Optional


class Speech(BaseModel):
    speaker: str
    content: str | bool
    supplemental: Optional[ModelResponse | list[ModelResponse]]


class SpeechType(Enum):
    PRE_FILLED = 1
    USER_INPUTTED = 2


class SpeechFormatEntry(BaseModel):
    speech_type: SpeechType
    prompt_tag: Optional[PromptTag]
    last_only_prompt_tag: Optional[PromptTag]
    expected_speaker: Optional[str]


class SpeechFormatType(Enum):
    DEFAULT_DEBATE = (
        auto(),
        lambda name, num_speeches, use_scratchpad, **kwargs: SpeechFormat.default_debate_format(
            name=name, num_speeches=num_speeches, use_scratchpad=use_scratchpad
        ),
    )
    DEFAULT_DEBATE_JUDGE = (
        auto(),
        lambda name, num_speeches, use_scratchpad, **kwargs: SpeechFormat.default_judge_format(
            name=name, num_speeches=num_speeches, use_scratchpad=use_scratchpad
        ),
    )
    DEFAULT_CONSULTANCY = (
        auto(),
        lambda name, num_speeches, use_scratchpad, **kwargs: SpeechFormat.default_consultancy_format(
            name=name, num_speeches=num_speeches, use_scratchpad=use_scratchpad
        ),
    )

    DEFAULT_CONSULTANCY_JUDGE = (
        auto(),
        lambda name, num_speeches, use_scratchpad, flipped, **kwargs: SpeechFormat.default_consultancy_judge_format(
            name=name, num_speeches=num_speeches, use_scratchpad=use_scratchpad, flipped=flipped
        ),
    )

    EMPTY_ROUND = (auto(), lambda name, **kwargs: SpeechFormat.empty_round_debater_format(name=name))

    EMPTY_ROUND_JUDGE = (auto(), lambda **kwargs: SpeechFormat.empty_round_judge_format())

    OPEN_DEBATE = (auto(), lambda name, **kwargs: SpeechFormat.open_debate_format(name=name, is_debate=True))

    OPEN_DEBATE_JUDGE = (
        auto(),
        lambda name, flipped, **kwargs: SpeechFormat.open_debate_format(name=name, flipped=flipped, is_debate=False),
    )

    def __init__(self, value: int, builder_func: Callable):
        self._value_ = value
        self.builder = builder_func

    def get_speech_format(self, **kwargs):
        return self.builder(**kwargs)


class SpeechFormatStructure(Enum):
    DEFAULT_DEBATE = (
        1,
        SpeechFormatType.DEFAULT_DEBATE,
        SpeechFormatType.DEFAULT_DEBATE_JUDGE,
        "Debate Prompt",
        2,
        False,
    )

    DEFAULT_CONSULTANCY = (
        2,
        SpeechFormatType.DEFAULT_CONSULTANCY,
        SpeechFormatType.DEFAULT_CONSULTANCY_JUDGE,
        "Consultancy Prompt",
        1,
        True,
    )

    EMPTY_ROUND = (3, SpeechFormatType.EMPTY_ROUND, SpeechFormatType.EMPTY_ROUND_JUDGE, "Empty Prompt", 0, True)

    OPEN_DEBATE = (4, SpeechFormatType.OPEN_DEBATE, SpeechFormatType.OPEN_DEBATE_JUDGE, "Open Debate Prompt", 1, True)

    def __new__(
        cls,
        value: int,
        debater_format: SpeechFormatType,
        judge_format: SpeechFormatType,
        default_prompt_name: str,
        num_participants: int,
        flip_position_order: bool,
    ):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.debater_format = debater_format
        obj.judge_format = judge_format
        obj.default_prompt_name = default_prompt_name
        obj.num_participants = num_participants
        obj.flip_position_order = flip_position_order
        return obj


class SpeechFormat:
    def __init__(self, name: str, tokens_per_speech: int = 300, num_speeches: int = 1):
        """
        A structure corresponding to the order of speeches and prompts that are expected to be delivered.

        Params:
            name: The name of the debater who is using this structure.
            tokens_per_speech: The length of each speech in tokens
            num_speeches: the number of speeches delivered by each debater/consultant in the round
        """
        self.progression = []
        self.name = name
        self.tokens_per_speech = tokens_per_speech
        self.num_speeches = num_speeches

    def add(
        self,
        speech_type: Optional[SpeechType] = None,
        prompt_tag: Optional[PromptTag] = None,
        last_only_prompt_tag: Optional[PromptTag] = None,
        expected_speaker: Optional[str] = None,
    ):
        """
        Adds an expected speech or command to the expected order. This does not add an actual speech -- it just tells
        the debater to expect a speech or command at a given point.

        Params:
            speech_type: The type of speech or command that is to be expected at this point.
            prompt_tag: If it is a command, this is the prompt tag corresponding to the command that is to be delivered at
                this point in the structure
            last_only_prompt_tag: If the prompt tag differs depending on if this is the last command the debater will hear
                before their next generation, then this is the tag they will use (defaults to prompt tag). An example is if
                the debater hears a command like "Please generate a speech..." before actually writing their speech, one might
                want this to show up in a transcript in the past tense like "This is the speech you gave" after one gives the
                speech.
            expected_speaker: The name of the speaker who is to deliver the speech.
        """
        speech_type = speech_type if speech_type else (SpeechType.PRE_FILLED if prompt_tag else SpeechType.USER_INPUTTED)
        self.progression.append(
            SpeechFormatEntry(
                speech_type=speech_type,
                prompt_tag=prompt_tag,
                last_only_prompt_tag=last_only_prompt_tag,
                expected_speaker=expected_speaker,
            )
        )
        return self

    def add_user_inputted_speech(self, expected_speaker: Optional[str] = None):
        """Adds a speech to the structure that is generated by an agent (debater or judge)"""
        return self.add(
            speech_type=SpeechType.USER_INPUTTED,
            prompt_tag=None,
            expected_speaker=expected_speaker if expected_speaker else self.name,
        )

    def add_format(self, speech_format: SpeechFormat, repeats: num_repetitions = 1):
        """Merges another speech format into this one"""
        for i in range(repeats):
            for speech_type, prompt_tag, last_only_prompt_tag, expected_speaker in speech_format:
                self.add(
                    speech_type=speech_type,
                    prompt_tag=prompt_tag,
                    last_only_prompt_tag=last_only_prompt_tag,
                    expected_speaker=expected_speaker,
                )
        return self

    def __iter__(self):
        self.idx = 0
        return self

    def __next__(self):
        if self.idx < len(self.progression):
            entry = self.progression[self.idx]
            self.idx += 1
            return entry.speech_type, entry.prompt_tag, entry.last_only_prompt_tag, entry.expected_speaker
        else:
            raise StopIteration

    @classmethod
    def default_debate_format(cls, name: str, num_speeches: int, use_scratchpad: bool, **kwargs) -> SpeechFormat:
        """Generates the order of speeches that the debater expects to receive"""
        opponent_name = (
            constants.DEFAULT_DEBATER_A_NAME
            if name == constants.DEFAULT_DEBATER_B_NAME
            else constants.DEFAULT_DEBATER_B_NAME
        )
        pre_debate = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.OVERALL_SYSTEM)
            .add(prompt_tag=PromptTag.DEBATER_SYSTEM)
            .add(prompt_tag=PromptTag.PRE_DEBATE)
        )

        scratchpad = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.PREVIOUS_DEBATER_SCRATCHPAD, last_only_prompt_tag=PromptTag.DEBATER_SCRATCHPAD)
            .add_user_inputted_speech(expected_speaker=name)
        )
        own_speech = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.PRE_PREVIOUS_SPEECH, last_only_prompt_tag=PromptTag.PRE_SPEECH)
            .add_user_inputted_speech(expected_speaker=name)
        )
        if use_scratchpad:
            own_speech = scratchpad.add_format(speech_format=own_speech)

        opponent_speech = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.PRE_OPPONENT_SPEECH)
            .add_user_inputted_speech(expected_speaker=opponent_name)
        )

        opening_statements = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.PRE_OPENING_SPEECH)
            .add_format(speech_format=own_speech)
            .add_format(speech_format=opponent_speech)
        )

        later_arguments = SpeechFormat(name).add_format(speech_format=own_speech).add_format(speech_format=opponent_speech)

        decision = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.JUDGE_DECISION_FOR_DEBATER)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        )

        return (
            SpeechFormat(name=name, tokens_per_speech=300, num_speeches=num_speeches)
            .add_format(speech_format=pre_debate)
            .add_format(speech_format=opening_statements)
            .add_format(speech_format=later_arguments, repeats=(num_speeches - 1))
            .add_format(speech_format=decision)
        )

    @classmethod
    def default_judge_format(cls, name: str, num_speeches: int, use_scratchpad: bool, **kwargs) -> SpeechFormat:
        pre_debate_speech_format = (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
            .add(prompt_tag=PromptTag.OVERALL_SYSTEM)
            .add(prompt_tag=PromptTag.JUDGE_SYSTEM)
            .add(prompt_tag=PromptTag.PRE_DEBATE_JUDGE)
        )

        argument_speech_format = (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
            .add(prompt_tag=PromptTag.PRE_DEBATER_A_SPEECH_JUDGE)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_DEBATER_A_NAME)
            .add(prompt_tag=PromptTag.PRE_DEBATER_B_SPEECH_JUDGE)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_DEBATER_B_NAME)
        )

        decision_speech_format = SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
        if use_scratchpad:
            decision_speech_format = (
                decision_speech_format.add(prompt_tag=PromptTag.POST_ROUND_JUDGE)
                .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
                .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
            )
        decision_speech_format = decision_speech_format.add(
            prompt_tag=PromptTag.POST_ROUND_JUDGE_WITHOUT_REASONING
        ).add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)

        return (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME, tokens_per_speech=150, num_speeches=num_speeches)
            .add_format(speech_format=pre_debate_speech_format)
            .add_format(speech_format=argument_speech_format, repeats=num_speeches)
            .add_format(speech_format=decision_speech_format)
        )

    @classmethod
    def default_consultancy_format(cls, name: str, num_speeches: int, use_scratchpad: bool, **kwargs) -> SpeechFormat:
        """Generates the order of speeches that the debater expects to receive"""

        pre_debate = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.OVERALL_SYSTEM)
            .add(prompt_tag=PromptTag.DEBATER_SYSTEM)
            .add(prompt_tag=PromptTag.PRE_DEBATE)
        )

        scratchpad = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.PREVIOUS_DEBATER_SCRATCHPAD, last_only_prompt_tag=PromptTag.DEBATER_SCRATCHPAD)
            .add_user_inputted_speech(expected_speaker=name)
        )
        own_speech = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.PRE_PREVIOUS_SPEECH, last_only_prompt_tag=PromptTag.PRE_SPEECH)
            .add_user_inputted_speech(expected_speaker=name)
        )
        if use_scratchpad:
            own_speech = scratchpad.add_format(speech_format=own_speech)

        opening_statements = (
            SpeechFormat(name).add(prompt_tag=PromptTag.PRE_OPENING_SPEECH).add_format(speech_format=own_speech)
        )

        later_arguments = SpeechFormat(name).add_format(speech_format=own_speech)

        decision = (
            SpeechFormat(name)
            .add(prompt_tag=PromptTag.JUDGE_DECISION_FOR_DEBATER)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        )

        return (
            SpeechFormat(name=name, tokens_per_speech=300, num_speeches=num_speeches)
            .add_format(speech_format=pre_debate)
            .add_format(speech_format=opening_statements)
            .add_format(speech_format=later_arguments, repeats=(num_speeches - 1))
            .add_format(speech_format=decision)
        )

    @classmethod
    def default_consultancy_judge_format(
        cls, name: str, num_speeches: int, use_scratchpad: bool, flipped: bool, **kwargs
    ) -> SpeechFormat:
        pre_debate_speech_format = (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
            .add(prompt_tag=PromptTag.OVERALL_SYSTEM)
            .add(prompt_tag=PromptTag.JUDGE_SYSTEM)
            .add(prompt_tag=PromptTag.PRE_DEBATE_JUDGE)
        )

        individual_round_format = (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
            .add(prompt_tag=PromptTag.PRE_DEBATER_A_SPEECH_JUDGE)  # this works even if the round is flipped
            .add_user_inputted_speech(
                expected_speaker=constants.DEFAULT_DEBATER_A_NAME if not flipped else constants.DEFAULT_DEBATER_B_NAME
            )
        )

        decision_speech_format = SpeechFormat(name=constants.DEFAULT_JUDGE_NAME)
        if use_scratchpad:
            decision_speech_format = (
                decision_speech_format.add(prompt_tag=PromptTag.POST_ROUND_JUDGE)
                .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
                .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
            )
        decision_speech_format = decision_speech_format.add(
            prompt_tag=PromptTag.POST_ROUND_JUDGE_WITHOUT_REASONING
        ).add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)

        return (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME, tokens_per_speech=150, num_speeches=num_speeches)
            .add_format(speech_format=pre_debate_speech_format)
            .add_format(speech_format=individual_round_format, repeats=num_speeches)
            .add_format(speech_format=decision_speech_format)
        )

    @classmethod
    def empty_round_debater_format(cls, name, **kwargs) -> SpeechFormat:
        return SpeechFormat(name=name)

    @classmethod
    def empty_round_judge_format(cls, **kwargs) -> SpeechFormat:
        return (
            SpeechFormat(name=constants.DEFAULT_JUDGE_NAME, tokens_per_speech=150)
            .add(prompt_tag=PromptTag.OVERALL_SYSTEM)
            .add(prompt_tag=PromptTag.JUDGE_SYSTEM)
            .add(prompt_tag=PromptTag.PRE_DEBATE_JUDGE)
            .add(prompt_tag=PromptTag.POST_ROUND_JUDGE_WITHOUT_REASONING)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        )

    @classmethod
    def open_debate_format(cls, name: str, flipped: bool = False, is_debate: bool = True, **kwargs) -> SpeechFormat:
        """Generates the order of speeches that the open debater expects to receive"""
        name_to_use = (
            name if is_debate else (constants.DEFAULT_DEBATER_A_NAME if not flipped else constants.DEFAULT_DEBATER_B_NAME)
        )
        return (
            SpeechFormat(name, tokens_per_speech=2)
            .add(prompt_tag=PromptTag.OVERALL_SYSTEM)
            .add(prompt_tag=PromptTag.DEBATER_SYSTEM)
            .add(prompt_tag=PromptTag.PRE_DEBATE)
            .add(prompt_tag=PromptTag.PRE_SPEECH)
            .add_user_inputted_speech(expected_speaker=name_to_use)
            .add_user_inputted_speech(expected_speaker=constants.DEFAULT_JUDGE_NAME)
        )
