
---
# Deep Reinforcement Learning for Feedback Control in a Collective Flashing Ratchet

[![arxiv](http://img.shields.io/badge/arXiv-2011.10357-B31B1B.svg)](https://arxiv.org/abs/2011.10357)
[![LICENSE](https://img.shields.io/github/license/kdkyum/RatchetDRL.svg)](https://github.com/kdkyum/RatchetDRL/blob/main/LICENSE)

Authors: Dong-Kyum Kim<sup>1</sup> and Hawoong Jeong<sup>1,2</sup><br>

<sup>1</sup> <sub>Department of Physics, KAIST</sub>
<sup>2</sup> <sub>Center for Complex Systems, KAIST</sub>

## Introduction

This repo contains source code for the runs in [Deep Reinforcement Learning for Feedback Control in a Collective Flashing Ratchet](https://arxiv.org/abs/2011.10357).

## Installation
```bash
git clone https://github.com/kdkyum/RatchetDRL
cd RatchetDRL
conda create -y --name ratchet python=3.7
conda activate ratchet
pip install -r requirements.txt
python -m ipykernel install --name ratchet
export PYTHONPATH='.'
```

## Usage

See option details by running the following command
```
python main.py --help
```

The training process is logged in `data/runs` directory. You can inspect the training process by tensorboard (run `tensorboard --logdir data/runs`).

```bash
# N=2, Smooth potential.
python main.py --env A --N 2

# N=2, Sawtooth potential.
python main.py --env B --N 2 

# N=4, PPO algorithm with DeepSets (ds) architecture, 10 time-steps are delayed. 
python main.py -a ds --env A_delay --N 4 --delay 10 

# N=4, PPO algorithm with RNN architecture, 10 time-stpes are delayed.
python main_rnn.py --env A_delay --N 4 --delay 10 
```

## Results
### Data
* `data/results` contains the all results (csv files) of the runs in the paper.
* `data/runs` contains the training logs and trained policy and value networks.

### Figures
* notebook files for plotting the figures in the paper.
  * [`data/Fig1-2-policy.ipynb`](data/Fig1-2-policy.ipynb): Figs. 1 and 2 (Decision boundaries for N=1 and N=2 cases).
  * [`data/Fig3-PPO_results.ipynb`](data/Fig3-PPO_results.ipynb): Fig. 3 (Results of PPO algorithm).
  * [`data/Fig1-2-policy.ipynb`](data/Fig1-2-policy.ipynb): Figs. 4 and 5 (Results of time-delayed feedback controls).
## Acknowledgement

This repository is built off the publicly released repository [openai/spinningup](https://github.com/openai/spinningup).

## License

This project following the MIT License.