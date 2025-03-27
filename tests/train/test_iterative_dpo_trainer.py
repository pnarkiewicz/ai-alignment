import pytest
import tempfile
from train.train_utils import TrainUtils
from models.deterministic_model import DeterministicModel

CONFIG_YAML = """
TestDeterministic: 
    model_name: stub_model
    target: debater
    llm_type: stub_llm
    training_hyperparameters:
        num_train_epochs: 2
        per_device_train_batch_size: 1
        gradient_accumulation_steps: 8
        optim: paged_adamw_32bit
        learning_rate: 2e-6
        max_grad_norm: 0.3
        warmup_ratio: 0.03
        lr_scheduler_type: constant
        peft_type: lora
        supplemental:
            judge_type: deterministic
            judge_fixed_output: "Debater A wins"
    logging_and_saving_config:
        logging_steps: 1
        output_dir: /fake/file/path
    dataset:
        dataset_type: quality 
"""


@pytest.fixture
def deterministic_config():
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".yaml") as tmpfile:
        tmpfile.write(CONFIG_YAML)
        tmpfile.flush()  # Ensures content is written
        config = TrainUtils.parse_config("TestDeterministic", tmpfile.name)
    return config


def test_judge_model_loading(deterministic_config):
    model = TrainUtils.load_judge_model(deterministic_config, is_local=False)
    assert isinstance(model, DeterministicModel)
