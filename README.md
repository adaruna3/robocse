# Robot Common-Sense Embedding (RoboCSE)
Code repository of RoboCSE. See [here](https://adaruna3.github.io/robocse/) to use an interactive tool the visualizes
the household domain knowledge and pre-trained embeddings.

## Pre-requisites
1. This repo has been tested for a system running Ubuntu 18.04 LTS, PyTorch (1.2.0), and 
hardware CPU or Nvidia GPU (GeForce GTX 1060 6GB or better).
2. For GPU functionality Nvidia drivers, CUDA, and cuDNN are required.

## Installation
All dependencies are installed to a virtual environment using `virtualenv` to protect your system's
current configuration. Install the virtual environment and dependencies by running `./setup_repo.sh`
in terminal. This script should only be executed ONCE for the life of the repo.

## Source the environment
You must source your environment each time it is deactivated. This is done via `source ./setup_env.sh`. You 
environment is sourced when `(py36_venv)` appears as the first part of the terminal prompt. You can unsource via
`deactivate`.

## Check install
After sourcing the environment, run `python`. Python version 3.6 should run. Next, check if `import torch` works.
Next, for GPU usage check if `torch.cuda.is_available()` is `True`. If all these checks passed, the installation should
be working. 

## Repo Conents
This repo contains the household domain knowledge used as input data, code to learn knowledge graph
embeddings, and pre-trained models developed for the [RoboCSE project](https://adaruna3.github.io/robocse/).

- Graph-embedding Models: [TrasnE](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf)
& [Analogy](http://proceedings.mlr.press/v70/liu17d.html)
- Datasets: The [THOR](./datasets/THOR_U) dataset was scraped from the simulator [AI2Thor](https://ai2thor.allenai.org/ithor/).
- Evaluation Conventions: Follow precedents & assumptions from knowledge graph embedding [community](http://papers.nips.cc/paper/5071-translating-embeddings-for-modeling-multi-relational-data.pdf).

## Explore the Input Data (Knowledge Graph)
Use the web visualization hosted [here](https://adaruna3.github.io/robocse/) to explore the dataset. Visualizations are
in the `Explore` tab.
    
## Train Models on the Knowledge Graph
The following scripts run the experiments presented in the submission. The final results of the scripts are CSV files
containing the metrics from the evaluations (total runtime < 10 minutes on GPU).

1. Train the models by running `./experiments/scripts/run_standard_setting_experiment_train.sh`.
2. Test the models by running `./experiments/scripts/run_standard_setting_experiment_test.sh`.

* After beginning a training program, you can check the progress of your training session by starting tensorboard in 
another terminal via `tensorboard --logdir=logger`. Remember to source the environment. As training progresses and the 
model achieves new best performance levels, model checkpooints are saved to `./models/checkpoints`.

## Hyper-paramter Tuning and Pre-trained Models:
We use Adagrad SGD to train the knowledge graph embeddings (TransE and Analogy). We tune all the hyper-parameters of 
knowledge graph embeddings simultaneously using grid search with the original knowledge graph (AI2Thor). For Analogy, 
we tune the learning rate {0.1,0.01,0.001}, negative sampling ratio {1,25,50,100}, and embedding hidden size dimensions 
(d_E/d_R) {25,50,100,200}. For TransE we also tune the hyper-parameter margin (gammea) {2,4,8}. The hyper-parameter 
settings and performance on the original knowledge graphs are shown below. Pre-treained models with the preivously 
mentioned hyper-parameters and performance metrics below are provided in [pre-trained](./models/pre-trained) folder of `models`.

|  Dataset |  Model  | Embedding Hidden Dim | Negative Sampling Ratio | Learning Rate | Margin | MRR% | Hits@10% |
|:--------:|:-------:|:--------------------:|:-----------------------:|:-------------:|:------:|:----:|:--------:|
|  AI2Thor |  Transe |          25          |            1            |      0.1      |   2.0  |  ~58 |    ~81   |
|  AI2Thor | Analogy |          100         |            50           |      0.1      |    -   |  ~64 |    ~86   |
