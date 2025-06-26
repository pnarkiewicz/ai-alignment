# TPML project: Debates of LLMs

This is a fork extending [nyu-debate-modeling](https://github.com/samuelarnesen/nyu-debate-modeling) project.

## Project info

This project explores debate-based protocols for achieving superalignment—the alignment of stronger AI models to the preferences of weaker models. Our work builds on and extends a prior research project and extend it based on feedback available on [OpenReview](https://openreview.net/forum?id=gAEEjGv5Oa).

We extend this work with additional debate protocols, training algorithms and datasets, evaluations and rewards, along with additional code optimizations.

## Simple local run for debugging purposes

Inside the repository root directory, create `.env` file:
``` txt
SRC_ROOT=/path/to/repo/
INPUT_ROOT=/path/to/repo/data/datasets/
```

Then execute the following commands to setup the environment.

``` bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
mkdir checkpoints
# ensure git lfs is installed
git clone https://huggingface.co/delphi-suite/v0-llama2-100k checkpoints/v0-llama2-100k
```

Now, you are ready to run a simple debug training with:
``` bash
python scripts/run_iterative_dpo.py --config="Local Train"
```


## Running training on athena cluster

The definition of singularity image with the required libraries is present in the `Singularity.def` file. The image based on this file is ready to run following the same setup as with local run. 

## Code structure

The most important or easily configurable parts of the code:

* Training algorithms are present in the `train` algorithms. They are mostly based on the internal implementations of HF `trl` library.
* Debaters and Judges extend a common `models.models.Model` abstract class. 
* Different trainning configurations (i.e. used algorithms, datasets, hyperparameters) are present inside `train/configs` directory.
* Prompts supplied to the Debaters / Judge are present in `prompts/configs`
