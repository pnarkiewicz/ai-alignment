# TPML project: Debates of LLMs

This is a fork extending [nyu-debate-modeling](https://github.com/samuelarnesen/nyu-debate-modeling) project.

## Project info

This project explores debate-based protocols for achieving superalignment—the alignment of stronger AI models to the preferences of weaker models. Our work builds on and extends a prior research project and extend it based on feedback available on [OpenReview](https://openreview.net/forum?id=gAEEjGv5Oa).

We extend this work with additional protocols, algorithms and datasets, along with additional code optimizations.

## Novelty


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

## Code structure



# Notes, TODO: delete



## Local DPO train
* Oparty na modelu: https://huggingface.co/delphi-suite/v0-llama2-100k
* Należy w pliku konfiguracyjnym `ai-ailgnment/train/configs/dpo_config.yaml` zaktualizować:
	* Ścieżkę do modelu `model_name`
 	* Ścieżkę do outputu `output_dir`
  * Najłatwiej to zrobić poprzez ściągnięcie repo przy pomocy gita 
    * By to zrobić trzeba wcześniej zainstalować [git-lfs](https://git-lfs.com/)
    * A następnie `git clone https://huggingface.co/delphi-suite/v0-llama2-100k checkpoints/v0-llama2-100k`
* Logi/wykresy lecą do `wandb`: wymaga zalogowania/tokena
* U mnie lokalnie leci ~4min, czasem efekty widać dużo szybciej. Starałem się zbić pamięć: raczej da się wyżyłować bardziej, ale to czasem prowadzi do błędów.
* W pliku `utils/constants.py` zdefiniowana zmienna `DEBUG`. Parę rzeczy zmieniam w kodzie, gdy jest ustawiona na `True`.


``` 
python scripts/run_iterative_dpo.py --config='Local Train'
```
## Setup notes - Piotrek:

* Należy stworzyć plik `.env` w bazowym folderze:
  ```
  SRC_ROOT=/ścieżka/do/repo/
  INPUT_ROOT=/ścieżka/do/repo/data/datasets/
  ```
  (ścieżki muszą kończyć się znakiem '/')
* Podstawowy test:
  ```
  python3.11 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  bash bash_scripts/basic_tests.sh
  ```

* Zbudowany obraz jest na athenie w folderze projektowym: `/net/pr2/projects/plgrid/plggaialignment/plgpikaminski/singularity_image.sif` (update: zła wersja biblioteki `trl`, trzeba zbić do `0.9.2`)

Changelog:
* Poluzowałem requirements.txt -- w oryginalnym repo były konflikty (mam nadzieję, że nic się potem nie wysypie)
* Wyrzuciłem jeden niezdefiniowany test z `bash_scripts/basic_tests.sh`
* Dodałem plik `Singularity.def` do budowania obrazu singularity. Virtual-env znajduje się w `/opt/venv`.
Jest mac-friendly (`BLIS_ARCH=generic`) i buduje się dosyć szybko ([tutorial od Janka](https://stackoverflow.com/questions/76457823/apptainer-on-macos)).
* Usunięte referencje do nieistniejących plików i dodany device `mps`

Problemy:
* Mam problem z puszczeniem treningu nawet na małych modelach (`opt-125m`) na macu (brak pamięci)
* Sporo bug'ów przy odpalaniu `scripts/run_ppo.py` -- kod częściowo wygląda na nieaktualny

TODO:
* `Judge` podczas treningu jest hardcoded na call'e do `OpenAI API`. Trzeba to sparametryzować

## Train - Przewodnik
* Zmienna `local` w configu odpowiada za testy. Nie jest to najlepsza nazwa i powinna być zmieniona na test, ale nie chciałem zmieniać od razu całego kodu.

## Tests
**pytest** could be particularly important when one's implement things with LLMs. One of most productive and error-prune way of "vibe-coding" is ask code-assistant for tests at the beggining.

Run all tests using:
```bash
pytest
```

## Git hooks
We use [pre-commit](https://pre-commit.com/) to manage git hooks.
To configure this extension one has to run
```sh
pre-commit install
```

## Athena Setup Notes - Running the Singularity/Apptainer image
### .env
Remember to create the `.env` file first with
```
SRC_ROOT=/PATH/TO/REPO/
INPUT_ROOT=/PATH/TO/REPO/data/datasets/
```

### 1. Start an **interactive** session (GPU-aware)
Allocate and run interactively a SLURM job
```bash
srun  --partition=plgrid-gpu-a100 --gres=gpu:1 --mem 32G --account=plgdebates2-gpu-a100 --time 0-01:00:00 --pty bash
```
Start an apptainer session
```bash
# If you’re already on a login node and have the image:
singularity shell \
  --nv                              # forward the NVIDIA driver to the container
  --home  /ABS/PATH/TO/PROJECT_DIR  # maps project into $HOME inside the image
  /ABS/PATH/TO/singularity.sif      # One can find jfpio images in /net/people/plgrid/plgjfpio/alignment_storage/singularity
```

### 2. Run one-off commands with **exec**

```bash
singularity exec --nv \
  --home /ABS/PATH/TO/PROJECT_DIR \
  /ABS/PATH/TO/singularity.sif \
  python scripts/run_debate.py \
    --configuration Simple_Test \
    --num_iters 1 --local --test --suppress_graphs --log_level INFO
```

Anything after the image path is executed **inside** the container, so you can
chain it with `srun`, `sbatch`, etc.


## Building an Apptainer Image on Mac (M1/M2/M3)

### Virtualization with Qemu (easier)
#### Requirements

- macOS with Apple Silicon (M1/M2/M3)
- [Lima](https://github.com/lima-vm/lima) installed
- QEMU installed (for x86\_64 emulation)

One can install it with 
```bash
brew install qemu lima
```

#### Steps

1. **Create an x86\_64 Lima Instance Using QEMU**

```bash
limactl create --vm-type=qemu --arch=x86_64 --name=apptainer template://apptainer
limactl start apptainer
```

This ensures the guest OS is running with x86\_64 architecture so the image will be compatible with AMD64-based HPC systems.

2. **Build the Image Inside the Instance**

```bash
limactl shell apptainer
apptainer build /tmp/lima/image.sif Singularity.def
```

This creates a `.sif` image compatible with x86\_64 systems.

### The newest version of apptainer build from source

#### Apptainer 1.4.0
The newest version of apptainer has `--arch` arg, so the x86\_64 emulation isn't needed.
`apptainer build --arch amd64 /tmp/lima/image.sif Singularity.def # --arch arg is available`

This part can be removed when Apptainer 1.4.0 will become available on [https://launchpad.net/~apptainer/+archive/ubuntu/ppa].
If so, only commands below will be needed.

```bash
limactl create --name=apptainer template://apptainer
limactl start apptainers
apptainer build --arch amd64 /tmp/lima/image.sif Singularity.def # --arch arg is available
```

#### Installing Apptainer 1.4.0 from Source with Cross-Build Support

This guide walks you through starting a Lima VM (with Rosetta support), installing required build dependencies, building Apptainer 1.4.0 from source, and using the new `--arch` flag to create an x86_64 container image—all while using temporary directories.

##### 1. Start the Lima VM with Rosetta

Launch the Lima VM (without needing persistence for this build):

```bash
limactl start template://apptainer --vm-type=vz --rosetta --name apptainer
```

###### 2. Install Build Dependencies

Inside the Lima shell, update the package list and install essential development tools. This ensures you have a C compiler, make, and other build essentials.

```bash
sudo apt update
sudo apt install -y build-essential
```

###### 3. (Optional) Install Go

Apptainer’s build requires Go (v1.22.7+). If it’s not already installed, do the following:

```bash
cd /tmp
wget https://go.dev/dl/go1.22.7.linux-arm64.tar.gz
sudo rm -rf /usr/local/go
sudo tar -C /usr/local -xzf go1.22.7.linux-arm64.tar.gz
export PATH=/usr/local/go/bin:$PATH
```

Verify installation:

```bash
go version
# Expected output: go version go1.22.7 linux/arm64
```

###### 4. Download and Extract the Apptainer 1.4.0 Source

Since these files are temporary, use the **/tmp** directory:

```bash
cd /tmp
wget https://github.com/apptainer/apptainer/releases/download/v1.4.0/apptainer-1.4.0.tar.gz
tar -xzf apptainer-1.4.0.tar.gz
cd apptainer-1.4.0
```

###### 5. Configure, Build, and Install Apptainer

Run the configuration script with your desired prefix, then compile and install:

```bash
./mconfig --prefix=/usr/local
make -C builddir
sudo make -C builddir install
```

Verify the installation:

```bash
apptainer --version
# Expected output: apptainer version 1.4.0
```

###### 6. Cross-Build an x86_64 Container Image

Assuming our definition file is named **Singularity.def**, build the image targeting the amd64 architecture:

```bash
apptainer build --arch amd64 /tmp/lima/singularity.sif Singularity.def
```

## 7. Transfer the Container Image

Assuming one's `.ssh/config` has the following entry
```
Host athena
    HostName athena.cyfronet.pl
    User user        
    ForwardAgent yes
    IdentityFile your_identity_file
```
One can copy the image to a remote HPC system using rsync:

```bash
rsync -avP /tmp/lima/sing.sif athena:/net/people/plgrid/$YOURUSERNAME/singularity.sif
```

---

This guide keeps your build environment clean by using **/tmp** for temporary files and ensures all necessary dependencies are installed. You now have an ARM-native Apptainer installation that can build x86_64 container images using the new `--arch` flag.

## Setup

Note: Given the current state of this project, this README will just give an overview of the code structure. It is not an introduction to the overall effort.
### Basic Setup
1. Pull down this package using `git clone`
2. Install the dependencies using `pip install -r requirements.txt`
3. Create an `.env` file with the following variables:
	`SRC_ROOT`: Contains the file path of the root of this code base
	`INPUT_ROOT`: Contains the file path to the directory where the dataset files are located
	`OPENAI_API_KEY`: OpenAI key (Optional, if one is using OpenAI for judging)
	`OPENAI_ORGANIZATION`: OpenAI organization (Optional, if one is using OpenAI for judging)
	`META_ACCESS_KEY`: Meta huggingface token, (Optional, if one is downloading a Llama model via HF)
	`ANTHROPIC_API_KEY`: Anthropic access key (Optional, if one is using Anthropic for judging)

### HPC Setup
1. Follow the HPC's guide to setup a Singularity container. Remember to install the dependencies in the virtual environment in that container.
2. Run `bash bash_scripts/hpc_setup.sh` from inside the Singularity container

## Tests

To run the tests, run `bash bash_scripts/basic_tests.sh`. To run an individual test, run `bash bash_scripts/operational_test.sh [TEST_NAME]`. To test the training code, run `bash bash_scripts/train_tests.sh`

### Scripts
The primary entrance points are in the `scripts` directory. Here is what each of the scripts are intended to do:
* **run_debate.py**: This kicks off a series of debate rounds. The configuration for those rounds (number of debaters, models used, number of speeches, use of scratchpad, decoding strategy, batching, etc.) is set in `experiments/configs`. Example Usage: `python3 ./scripts/run_debate.py --num_iters=50 --log_level='DEBUG' --configuration='Test'`
* **run_iterative_dpo.py**: Kicks off a DPO training run. The configuration for the training run is set in `train/configs/dpo_config.yaml`. Example usage: `python3 ./scripts/run_iterative_dpo.py --configuration='Test'`
* **run_sft.py**: Kicks off a SFT training run. The configuration for the training run is set in `train/configs/sft_config.yaml`. Example usage: `python3 ./scripts/run_sft.py --configuration='Test'`

## Code Structure

### High-Level Summary

One can use this package to train models or to run experiments (aka validation or inference). The key modules are as follows:

1. **Data:** This module provides a unified interface for loading, parsing, and interacting with the debate datasets. 
2. **Prompts:** This modules contains a configuration file and parser where you define the prompts you expose to the debaters and judge. It interacts with the `data` module to fill out the prompts with actual values.
3. **Models:** This module contains a unified interface for the resources that can generate text. Each model can wrap around an open LLM (e.g. Llama 2), an API (e.g. OpenAI's GPT4), or any other mechanism for generating text (e.g. it is convenient, for testing purposes, to have a model that generates random text). 
4. **Debate:** This module defines the core debate-specific abstractions. That includes the abstraction for an agent (i.e. a `Debater` or `Judge`), a `Transcript`, a speech order, and a `DebateRound`. The agents maintain an internal transcript and wrap around a `Model` in order to generate text.
5. **Experiment:** This contains the logic for running experiments. A loader read a config file and creates a series of `DebateRound` objects, whose results are then tallied, logged, and displayed.
6. **Train:** This modules contains all the logic for training a model. It interacts with the `prompt`, `data`, `models` and `debate` modules to consistency across input and output formats.
7. **Scripts:** This is the entrance point for running experiments and training models.

### Data
This module handles the loading of different datasets. The data loaded in these datasets is used for the questions that are debated in the experiments as well as the training signal for the training scripts. The datasets are as follows:
* **Quality Debates**: These are the debates that were run as part of the NYU Human Debate experiments.
* **Quality**: These are the full set of hard questions from the QuALITY dataset.

### Debate
This module controls the primary abstractions for actually running experiments. The main abstractions are as follows:
* **Debate Round**: This is a structure containing two debaters and a judge. Each debater or judge generates speeches and collect their own transcripts. The judge is the main coordinator for the round, so the primary role of the `DebateRound` object is to pass information (speeches) from the different debaters and to report the winner at the end of the round.
* **Judge**: This agent asks questions, renders verdicts, and decides who the next speaker is. It uses a `Model` to generate text. It receives a `SpeechFormat` argument that determines the order of speeches that it expects. There is a child `BranchedJudge` class for the use in branched rounds.
* **Debater**: This wraps around a `Model` class, which it uses to generate speeches when called. There is a child `BestOfNDebater` class for Best-of-N rounds (when it has to generate multiple speeches). It receives a `SpeechFormat` argument that determines the order of speeches that it expects.
* **Transcript**: Each participant (judge and debater) maintains a `Transcript` object that tracks the round so far. It can convert into either a `ModelInput` format that can be used by a `Model` or a string that can be written to the logs.
* **Speech Format**: This defines the order of speeches. It has default options for both debate and consultancy. These get passed to the debaters and judge so that they know whose turn it is to speak.

### Experiments
This module controls the configuration for running experiments (aka debate rounds) as well as the metrics collection for those experiments. See `experiments/configs/example_experiment.yaml` for a detailed explanation of each potential configuration option.
* **experiment_loader.py**: This reads from the configuration set in `experiments/configs` and constructs a set of `DebateRound` objects.
* **quotes_collector.py**: This handles the metrics collections for the metrics about quotation usage.
* **results_collector.py**: This handles the overall metric collection.

### Models
This module also contains the `Model` abstraction that is used to generate text. The models are as follows:
* **Deterministic Model**: This model just outputs the same text over and over again. This is useful when testing.
* **LLM Model**: This model generates text by invoking a model from Huggingface (yes I know the "M" in LLM stands for "model" but "ll_model" looked strange). It has child classes for different flavors of Huggingface models that have different input formats. At the moment, those two implementing classes are for Llama and Mistral models. This also contains logic for implementing a Linear Probe judge (aka a judge that adds a linear layer on top of the activations of another model), however this is less tested.
* **Offline Model**: This model just repeats the text from a text transcript that it is provided. This is useful if one wants to re-evaluate the debate with a different judge than the one originally used.
* **OpenAI Model**: This model generates text by calling OpenAI.
* **Anthropic Model**: This model generates text by calling Anthropic's Claude model.
* **Random Model**: This model generates random strings of text. It is useful when testing.
* **Served Model**: This model makes requests to a localhost destination where it expects a model to be hosted (this can speed up inference dramatically if the model is hosted using an inference engine).
* **Human Model**: This model outputs real text from the corresponding human debate transcripts. It only works if one is sampling questions from the real human debates. (Not recommended)

### Prompts
This module controls the prompts that are used while conducting experiments or training. There is a parser for both normal prompts and 'dynamic prompts' (which are just prompts that change depending on certain attributes). The actual prompt language can be found in `prompts/configs`

### Train
This module controls the logic for finetuning the models using DPO, PPO, or supervised finetuning. The specific classes are:
* **DPO Trainer**: This is a class for training models using DPO
* **PPO Trainer**: This is a class for training models using DPO
* **SFT Trainer**: This is a class for training a model to debate via supervised finetuning.
* **Row Converter**: This is a utility for converting a transcript into the format expected by the trainers. To do so, it interacts with the `debate`, `prompts`, and `data` abstractions.
* **Train Utils**: Additional utilities for loading models and datasets, specifically for training.

## Potential Uses

### Running a Tournament (Data Generation / Validation)
1. Create a new config entry under `experiments/configs`. If you're running this locally, add the entry under `test_experiment.yaml` and if you're running it remotely, add it under `standard_experiment.yaml`. 
2. To kick off the new tournament, run `python python3 ./scripts/run_debate.py --num_iters=Number-of-Iterations --configuration='Your-New-Configuration-Name`. If you're running it locally, you'll need to a `--test` flag so it knows to look in the right place.

**Examples:**
All of the following examples can be found under `experiments/configs/standard_experiment.yaml` unless otherwise specified.
* **Debate - Data Generation**: See the entry for `Data Generation - Llama3 - MultiRound - HalfBranched - FullTrain - DPO`. This demonstrates a single model with branched rounds.
* **Consultancy - Data Generation**: See the entry for `Data Generation - Llama3 - MultiRound - HalfBranched - FullTrain - Consultancy`
* **Debate - Self Play Validation**: See the entry for `val - experiment debate - 16`.
* **Consultancy - Self Play Validation**: See the entry for `val - experiment consultant - 16`
* **Debate - Cross Play Tournament**: See the entry for `val - cross-play - 0 - 176 - 464`. This config covers a small slice of a broader round-robin tournament. (For ease in parallelizing the cross-play tournament across multiple machines, we split up the round robin into small chunks -- note that, while only three models are used in this particular run, all of the other models are still defined so that the proper rounds are assigned to each model)
* **Debate - Offline Judging**: See `Offline_Debate_Test` under `test_experiment.yaml`.
* **Consultancy - Offline Judging**: See `MultiTurn_Consultancy_Offline_Test` under `test_experiment.yaml`. Note that `alternate` is set to True (this is needed only for offline consultancies).
* **Converting Single Consultancy to Double Consultancy**: See the entry for `Consultancy_Double_Test` under `test_experiment.yaml`. Note that the speech structure is `default_debate` and that `convert_to_double_consultancy` is True.

### Training a New Model
1. Depending on how you want to train your model (SFT, DPO, PPO, etc.), add a new entry in the associated config that can be found under `train/configs/[dpo/ppo/sft]_config.yaml`.
2. Kick off your training run by running the associated script in the `scripts` directory (e.g. `python scripts/run_sft.py --config="YourNewConfigName"`).

**Examples:**
1. **DPO Training** See `Iterative - FullTrainDebateLowLR - 77` under `train/configs/dpo_config.yaml`. Note how multiple existing datasets are concatenated together to train a single model.
2. **SFT Training** See `Train - Llama3 - Human and GPT` under `train/configs/sft_config.yaml`

### Adding a New Dataset
You might want to create a new dataset if you're using a data source other than QuALITY. Here are the steps to add a new dataset:
1. Create a new directory under `data/datasets/` and add your file(s) there.
2. Under the `data` directory, create a python file called `[your_dataset_name]_loader.py`. In that file, you will define two classes, a `[YourDatasetName]Loader` and `[YourDatasetName]Dataset`. The loader should just parse your dataset and pass it in to the dataset constructor. The dataset itself will split out the data into the rows that you want, following the interface defined in `data/dataset.py`. The file `data/quality_loader.py` is also a good example that one can follow (although it has some extra deduplication logic you may not need).
3. Under `data/dataset.py`, add a new enum corresponding to your new dataset type.
4. Under `data/loader_utils.py`, add a new conditional to support creating your new dataset.
5. Under `data/__init__.py`, import your new dataset.

Now you should be good to reference your new dataset from a train or experiment config file.

### Adding New Prompts
1. Go to `prompts/configs/prompts.yaml` and add a new entry corresponding to your new set of prompts. Feel free to follow the examples in that file -- the two sets of existing prompts are called `Debate Prompt` and `Consultancy Prompt`.
2. If your new prompt uses any unique names that do not already exist in the existing prompts (the 'names' are the titles of each individual sub-entry such as `overall_system` or `pre_opening_speech`), then go to `prompts/parser.py` and add that title as a new enum under `PromptTag`.
3. If you require filling in a new kind of value (currently, we support filling out prompts with a debater name, opponent name, position, opponent position, topic/question, and the background text), then add that new kind of value under `PromptConfig` in `prompts/parser.py`. This new value will be inserted into the prompts as long as the exact key is referenced inside angular brackets (e.g. if you reference `<NAME>` in your prompts, then `<NAME>` will be replaced with the value associated with `name` in the PromptConfig.) 

### Creating New Speech Orders
You might want to create new speech order if you do not want the speeches to be delivered in the same format as we have previously set.
1. Under `debate/speech_format.py`, create a new method in the `SpeechFormat` object, following the example of `default_debate_format()` and `default_judge_format()`. 
2. Also under `debate/speech_format.py`, create a new enum under `SpeechFormatStructure.`
3. Also under `debate/speech_format.py`, create a new enum under `SpeechFormatType.`

Now you can reference your new speech orders in your configs by using the name you gave under `SpeechFormatType`.

### Using a new open-weight LLM
We currently support generating speeches using Mistral or Llama. If you want to add a different type from Huggingface, you should do the following:
1. Under `models/llm_model.py`, create a new class that inherits from `LLModel`. At a minimum, it just needs to define the instruction prefixes and suffixes. See the `LlamaModel` class as a good example. It also should implement a `copy()` method.
2. Also under `models/llm_model.py`, create a new enum under `LLMType`.

Now you should be free to reference your new model type in your configs by using the name you defined as the `LLMType` enum.
