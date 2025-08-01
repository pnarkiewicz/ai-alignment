from __future__ import annotations

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer
from models.model import Model, ModelInput, ModelResponse, SpeechStructure
from prompts import RoleType
from utils.constants import DEBUG, DEFAULT_DEBATER_A_NAME, DEFAULT_DEBATER_B_NAME

GLOBAL_MODEL = None
GLOBAL_TOKENIZER = None

class HFWrapperJudge(Model):
    def __init__(
        self,
        alias: str,
        is_debater: bool = False,
        judge_model: str = "meta-llama/Llama-3.2-3B-Instruct",
    ):
        """
        Wraps a LLaMA model and uses log-probabilities to pick a winner between two completions.
        Can only be used as a judge.

        Args:
            alias: Identifier name for the model
            is_debater: Must be False. This model is only for judging.
            model_name: Hugging Face model name or path to LLaMA weights
        """
        super().__init__(alias=alias, is_debater=False)
        self.train_step = 0
        if is_debater:
            raise Exception("ArbitraryAttributeModel only supports judge mode")

        self.model_name = judge_model
        global GLOBAL_MODEL, GLOBAL_TOKENIZER

        if GLOBAL_MODEL is None or GLOBAL_TOKENIZER is None:
            GLOBAL_TOKENIZER = AutoTokenizer.from_pretrained(judge_model)
            model = AutoModelForCausalLM.from_pretrained(judge_model)
            model.eval()
            if torch.cuda.is_available():
                model.cuda()
            GLOBAL_MODEL = model

        self.tokenizer = GLOBAL_TOKENIZER
        self.model = GLOBAL_MODEL

    def _get_log_prob(self, prompt: str, continuation: str) -> float:
        input_text = prompt + continuation
        inputs = self.tokenizer(input_text, return_tensors="pt")
        target_ids = self.tokenizer(continuation, return_tensors="pt").input_ids

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            target_ids = target_ids.cuda()

        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            logits = outputs.logits

        # Only consider log probs for the continuation portion
        continuation_start = inputs["input_ids"].shape[1] - target_ids.shape[1]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

        selected = torch.gather(
            log_probs[:, continuation_start - 1 : -1, :],
            2,
            target_ids.unsqueeze(-1),
        ).squeeze(-1)

        return selected.sum().item()

    def predict(
        self,
        inputs: list[list[ModelInput]],
        max_new_tokens=250,
        speech_structure: SpeechStructure = SpeechStructure.DECISION,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> list[ModelResponse]:
        if speech_structure != SpeechStructure.DECISION:
            raise Exception("Only supports decision speech structure")

        if len(inputs) > 1 and num_return_sequences > 1:
            raise Exception("Cannot use both batched inputs and multiple return sequences")

        results = []
        for debate_inputs in inputs:
            prompt = "\n".join([mi.content for mi in debate_inputs])
            choice_a = DEFAULT_DEBATER_A_NAME
            choice_b = DEFAULT_DEBATER_B_NAME

            log_prob_a = self._get_log_prob(prompt, choice_a)
            log_prob_b = self._get_log_prob(prompt, choice_b)

            prob_a = torch.softmax(torch.tensor([log_prob_a, log_prob_b]), dim=0)[0].item()
            prob_b = 1 - prob_a
            decision = choice_a if prob_a >= prob_b else choice_b

            print("a = ", prob_a)
            if DEBUG:
                wandb.log({ "train_prob/step": self.train_step,
                            "train_prob/prob_a": prob_a,
                            "train_prob/prob_b": prob_b})
                self.train_step += 1

            results.append(
                ModelResponse(
                    decision=decision,
                    probabilistic_decision={choice_a: prob_a, choice_b: prob_b},
                    prompt=prompt,
                )
            )

        return results

    def copy(self, alias: str, is_debater: bool | None = None, **kwargs):
        return HFWrapperJudge(
            alias=alias,
            is_debater=is_debater if is_debater is not None else False,
            judge_model=self.model_name,
        )
