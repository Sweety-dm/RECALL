# Replay-enhanced Continual Reinforcement Learning

This is the code and experimental results for the paper `Replay-enhanced Continual Reinforcement Learning`. 
It proposes a method called `RECALL` for replay-based continual reinforcement learning, which is experimentally
validated on the [Continual World](https://github.com/awarelab/continual_world) benchmark and accepted as is in TMLR 2023.

See also the [paper](https://openreview.net/pdf?id=91hfMEUukm) and
the [OpenReview Website](https://openreview.net/forum?id=91hfMEUukm).


# Installation

First, you'll need [MuJoCo](http://www.mujoco.org/) simulator. Please follow
the [instructions](https://github.com/openai/mujoco-py#install-mujoco)
from `mujoco_py` package. As MuJoCo has been made freely available, you can obtain a free
license [here](https://www.roboti.us/license.html).

Next, go to the `metaworld` directory of this repo and run

`pip install .`

to install MetaWorld package that Continual World heavily relies on. 
Alternatively, if you want to install in editable mode, run

`pip install -e .`


# Running

You can run continual learning experiments with `run_cl.py` scripts.

To see available script arguments, run with `--help` option, e.g.

`python3 run_cl.py --help`

Commands to run experiments that reproduce main results from the paper can be found
in `run_cl.sh`. Because of number of different runs that
these files contain, it is infeasible to just run it in sequential manner. We hope though that these files will be
helpful because they precisely specify what needs to be run.


# Producing result tables and plots

After you've run experiments and you have saved logs, you can run the script to produce result tables and plots.
The commands can be found in `produce_results.sh`.

In these commands, respective arguments should be replaced for paths to directories containing logs from continual
learning and baseline (single-task) experiments. Each of these should be a directory inside which there are multiple experiments, for different 
methods and/or seeds.


# Acknowledgements

The implementation of SAC, EWC, PackNet, and Perfect Memory used in our code comes from
[Continual World](https://github.com/awarelab/continual_world).
