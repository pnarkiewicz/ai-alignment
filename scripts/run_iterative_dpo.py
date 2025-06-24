from script_utils import ScriptUtils, TrainType

import wandb

ScriptUtils.setup_script()

from train import IterativeDirectPreferenceTrainer, TrainUtils

args = ScriptUtils.get_args()
script_config = ScriptUtils.get_training_run_script_config(args, train_type=TrainType.DPO)


config = TrainUtils.parse_config(config_name=script_config.config_name, config_filepath=script_config.config_filepath)

wandb.init(project="huggingface")
wandb.define_metric("train/*", step_metric="train/step")
wandb.define_metric("eval/*", step_metric="eval/step")
wandb.define_metric("eval_first_true/*", step_metric="eval_first_true/step")
wandb.define_metric("eval_first_false/*", step_metric="eval_first_false/step")
wandb.define_metric("eval_second_true/*", step_metric="eval_second_true/step")
wandb.define_metric("eval_second_false/*", step_metric="eval_second_false/step")
wandb.define_metric("train_prob/*", step_metric="train_prob/step")

trainer = IterativeDirectPreferenceTrainer(config=config, smooth=True, is_local=args.test)
epoch_size = (
    config.training_hyperparameters.supplemental.get("epoch_size", 2048)
    if config.training_hyperparameters.supplemental
    else 2048
)

if not args.test:
    trainer.train(epoch_size=epoch_size)
else:
    samples = trainer.get_samples(start_idx=0, epoch_size=epoch_size)
