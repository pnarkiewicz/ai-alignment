from __future__ import annotations

import random
import re

import utils.constants as constants
import wandb
from models.model import Model, ModelInput, ModelResponse, SpeechStructure
from prompts import RoleType
from utils.constants import DEBUG


class ArbitraryAttributeModel(Model):
    def __init__(
        self,
        alias: str,
        is_debater: bool = False,
        feature: str | None = None,
        **kwargs,
    ):
        """
        An ArbitraryAttributeModel model picks up on an arbitrary but deterministic feature.
        Can be used only for judging. Useful for testing.

        Args:
            alias: string that identifies the model for metrics and deduplication
            is_debater: boolean indicating whether the model is a debater (true) or judge (false)
            feature: the attribute to count when judging
        """
        super().__init__(alias=alias, is_debater=is_debater)
        if is_debater:
            raise Exception("ArbitraryAttributeModel only supports judge mode")
        self.feature = feature or "quote"  # TODO: change
        self.evaluate = False
        self.train_step = 0

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens=250,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[str]:
        """
        Generates a list of texts in response to the given input.

        Args:
            inputs: A list of list of model inputs. Each ModelInput corresponds roughly to one command,
                a list of ModelInputs corresponds to a single debate (or entry in a batch), and so the
                list of lists is basically a batch of debates. Since the model will return a random
                response no matter what, the content of the input does not matter.
            max_new_tokens: The total number of new tokens to generate.
            speech_structure: The format that the answer is expected to be in. Option includes "open-ended"
                (which is just free text) and "decision" (which means a boolean is expected)
            num_return_sequences: The number of responses that the model is expected to generate. If a batch
                size of >1 is passed in, then this value will be overridden by the batch size (so you cannot
                have both num_return_sequences > 1 and len(inputs) > 1)

        Returns:
            A list of text, with one string for each entry in the batch (or for as many sequences are specified
            to be returned by num_return_sequences).

        Raises:
            Exception: Raises Exception if num_return_sequences > 1 and len(inputs) > 1
        """

        def score_speeches(
            model_inputs: list[ModelInput],
        ) -> tuple[str, tuple[float, float]]:
            speeches = [speech for speech in filter(lambda x: x.role == RoleType.USER, model_inputs)]

            a_speech_pre = re.search(
                "This is what Debater_A said during their speech.(.*)This is what Debater_B said during their speech",
                speeches[-1].content,
                flags=re.DOTALL,
            )

            a_speech = a_speech_pre.group(1).strip()
            b_speech_pre = re.search(
                "This is what Debater_B said during their speech.(.*)Now it is the time for a decision",
                speeches[-1].content,
                flags=re.DOTALL,
            )

            b_speech = b_speech_pre.group(1).strip()

            a_score = a_speech.count(self.feature)
            b_score = b_speech.count(self.feature)

            if not self.evaluate:
                wandb.log({"feature_count": a_score + b_score}, step=self.train_step)
                wandb.log({"generated_length": len(a_speech) + len(b_speech)}, step=self.train_step)
                wandb.log({"feature_frac": (a_score + b_score + 1e-5) / (len(a_speech) + len(b_speech) + 1e-5)}, step=self.train_step)
                wandb.log({f"A score {self.evaluate=}": a_score}, step=self.train_step)
                wandb.log({f"B score {self.evaluate=}": b_score}, step=self.train_step)
                self.train_step += 1

            random_val = random.random()
            epsilon = 1e-2
            rand1 = random.random() * epsilon
            rand2 = random.random() * epsilon
            if a_score + b_score > 0:
                if DEBUG:
                    a_score += rand1 + 1e-5
                    b_score += rand2 + 1e-5
                ret_val = (
                    (constants.DEFAULT_DEBATER_A_NAME if a_score >= b_score else constants.DEFAULT_DEBATER_B_NAME),
                    (
                        a_score / (a_score + b_score),
                        b_score / (a_score + b_score),
                    ),
                )
            else:
                ret_val = (
                    (constants.DEFAULT_DEBATER_A_NAME if random_val <= 0.5 else constants.DEFAULT_DEBATER_B_NAME),
                    (
                        0.5 + rand1,
                        0.5 - rand1,
                    ),
                )

            return ret_val

        if speech_structure != SpeechStructure.DECISION:
            raise Exception("ArbitraryAttributeModel only supports making decisions")

        if len(inputs) > 1 and num_return_sequences > 1:
            raise Exception(
                f"Length of input ({len(inputs)}) and num_return_sequences ({num_return_sequences}) cannot "
                "both be greater than 1."
            )

        decisions = []
        for i in range(len(inputs)):
            decision, (a_odds, b_odds) = score_speeches(inputs[i])
            decisions.append(
                ModelResponse(
                    decision=decision,
                    probabilistic_decision={
                        constants.DEFAULT_DEBATER_A_NAME: a_odds,
                        constants.DEFAULT_DEBATER_B_NAME: b_odds,
                    },
                    prompt="\n".join([model_input.content for model_input in inputs[i]]),
                )
            )
        return decisions

    def copy(self, alias: str, is_debater: bool | None = None, **kwargs):
        """Generates a deepcopy of this model"""
        return ArbitraryAttributeModel(
            alias=alias,
            is_debater=is_debater if is_debater is not None else False,
            feature=self.feature,
        )
