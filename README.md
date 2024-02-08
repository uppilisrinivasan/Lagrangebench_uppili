# LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite

[![Static Badge](https://img.shields.io/badge/docs-red?style=for-the-badge&logo=readthedocs&link=https%3A%2F%2Flagrangebench.readthedocs.io%2Fen%2Flatest%2Findex.html)](https://lagrangebench.readthedocs.io/en/latest/index.html)
[![Static Badge](https://img.shields.io/badge/arxiv-blue?style=for-the-badge&logo=arxiv&link=https%3A%2F%2Farxiv.org%2Fabs%2F2309.16342)](https://arxiv.org/abs/2309.16342)

## Installation
### Standalone library
Install the core `lagrangebench` library from PyPi as
```bash
pip install lagrangebench --extra-index-url=https://download.pytorch.org/whl/cpu
```

Note that by default `lagrangebench` is installed without JAX GPU support. For that follow the instructions in the [GPU support](#gpu-support) section.

### Clone
Clone this GitHub repository
```bash
git clone https://github.com/tumaer/lagrangebench.git
cd lagrangebench
```

Install the dependencies with __Poetry (>=1.6.0)__
```
poetry install --only main
```

Alternatively, a requirements file is provided. It directly installs the CUDA version of JAX.
```
pip install -r requirements_cuda.txt
```
For a CPU version of the requirements file, one could use `docs/requirements.txt`.

### GPU support
To run JAX on GPU, follow the [Jax CUDA guide](https://github.com/google/jax#pip-installation-gpu-cuda-installed-via-pip-easier), or in general run
```bash
pip install --upgrade jax[cuda11_pip]==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# or, for cuda 12
pip install --upgrade jax[cuda12_pip]==0.4.20 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### MacOS
Currently, only the CPU installation works. You will need to change a few small things to get it going:
- Clone installation: in `pyproject.toml` change the torch version from `2.1.0+cpu` to `2.1.0`. Then, remove the `poetry.lock` file and run `poetry install --only main`.
- Configs: You will need to set `f64: False` and `num_workers: 0` in the `configs/` files.

Although the current [`jax-metal==0.0.5` library](https://pypi.org/project/jax-metal/) supports jax in general, there seems to be a missing feature used by `jax-md` related to padding -> see [this issue](https://github.com/google/jax/issues/16366#issuecomment-1591085071).

## Usage
### Standalone benchmark library
A general tutorial is provided in the example notebook "Training GNS on the 2D Taylor Green Vortex" under `./notebooks/tutorial.ipynb` on the [LagrangeBench repository](https://github.com/tumaer/lagrangebench). The notebook covers the basics of LagrangeBench, such as loading a dataset, setting up a case, training a model from scratch and evaluating its performance.

### Running in a local clone (`main.py`)
Alternatively, experiments can also be set up with `main.py`, based on extensive YAML config files and cli arguments (check [`configs/`](configs/)). By default, the arguments have priority as: 1) passed cli arguments, 2) YAML config and 3) [`defaults.py`](lagrangebench/defaults.py) (`lagrangebench` defaults).

When loading a saved model with `--model_dir` the config from the checkpoint is automatically loaded and training is restarted. For more details check the [`experiments/`](experiments/) directory and the [`run.py`](experiments/run.py) file.

**Train**

For example, to start a _GNS_ run from scratch on the RPF 2D dataset use
```
python main.py --config configs/rpf_2d/gns.yaml
```
Some model presets can be found in `./configs/`.

If `--mode=all`, then training (`--mode=train`) and subsequent inference (`--mode=infer`) on the test split will be run in one go.


**Restart training**

To restart training from the last checkpoint in `--model_dir` use
```
python main.py --model_dir ckp/gns_rpf2d_yyyymmdd-hhmmss
```

**Inference**

To evaluate a trained model from `--model_dir` on the test split (`--test`) use
```
python main.py --model_dir ckp/gns_rpf2d_yyyymmdd-hhmmss/best --rollout_dir rollout/gns_rpf2d_yyyymmdd-hhmmss/best --mode infer --test
```

If the default `--out_type_infer=pkl` is active, then the generated trajectories and a `metricsYYYY_MM_DD_HH_MM_SS.pkl` file will be written to the `--rollout_dir`. The metrics file contains all `--metrics_infer` properties for each generated rollout.

## Datasets
The datasets are hosted on Zenodo under the DOI: [10.5281/zenodo.10021925](https://zenodo.org/doi/10.5281/zenodo.10021925). When creating a new dataset instance, the data is automatically downloaded. Alternatively, to manually download them use the `download_data.sh` shell script, either with a specific dataset name or "all". Namely
- __Taylor Green Vortex 2D__: `bash download_data.sh tgv_2d datasets/`
- __Reverse Poiseuille Flow 2D__: `bash download_data.sh rpf_2d datasets/`
- __Lid Driven Cavity 2D__: `bash download_data.sh ldc_2d datasets/`
- __Dam break 2D__: `bash download_data.sh dam_2d datasets/`
- __Taylor Green Vortex 3D__: `bash download_data.sh tgv_3d datasets/`
- __Reverse Poiseuille Flow 3D__: `bash download_data.sh rpf_3d datasets/`
- __Lid Driven Cavity 3D__: `bash download_data.sh ldc_3d datasets/`
- __All__: `bash download_data.sh all datasets/`


### Notebooks
We provide three notebooks that show LagrangeBench functionalities, namely:
- [`tutorial.ipynb`](notebooks/tutorial.ipynb) with a general overview of LagrangeBench library, with training and evaluation of a simple GNS model,
- [`datasets.ipynb`](notebooks/datasets.ipynb) with more details and visualizations on the datasets, and
- [`gns_data.ipynb`](notebooks/gns_data.ipynb) showing how to train models within LagrangeBench on the datasets from the paper [Learning to Simulate Complex Physics with Graph Networks](https://arxiv.org/abs/2002.09405).

## Directory structure
```
📦lagrangebench
 ┣ 📂case_setup     # Case setup manager
 ┃ ┣ 📜case.py      # CaseSetupFn class
 ┃ ┣ 📜features.py  # Feature extraction
 ┃ ┗ 📜partition.py # Alternative neighbor list implementations
 ┣ 📂data           # Datasets and dataloading utils
 ┃ ┣ 📜data.py      # H5Dataset class and specific datasets
 ┃ ┗ 📜utils.py
 ┣ 📂evaluate       # Evaluation and rollout generation tools
 ┃ ┣ 📜metrics.py
 ┃ ┗ 📜rollout.py
 ┣ 📂models         # Baseline models
 ┃ ┣ 📜base.py      # BaseModel class
 ┃ ┣ 📜egnn.py
 ┃ ┣ 📜gns.py
 ┃ ┣ 📜linear.py
 ┃ ┣ 📜painn.py
 ┃ ┣ 📜segnn.py
 ┃ ┗ 📜utils.py
 ┣ 📂train          # Trainer method and training tricks
 ┃ ┣ 📜strats.py    # Training tricks
 ┃ ┗ 📜trainer.py   # Trainer method
 ┣ 📜defaults.py    # Default values
 ┗ 📜utils.py
```


## Citation
The paper (at NeurIPS 2023 Datasets and Benchmarks) can be cited as:
```bibtex
@inproceedings{toshev2023lagrangebench,
    title      = {LagrangeBench: A Lagrangian Fluid Mechanics Benchmarking Suite},
    author     = {Artur P. Toshev and Gianluca Galletti and Fabian Fritz and Stefan Adami and Nikolaus A. Adams},
    year       = {2023},
    url        = {https://arxiv.org/abs/2309.16342},
    booktitle  = {37th Conference on Neural Information Processing Systems (NeurIPS 2023) Track on Datasets and Benchmarks},
}
```

The associated datasets can be cited as:
```bibtex
@dataset{toshev_2023_10021926,
  author       = {Toshev, Artur P. and Adams, Nikolaus A.},
  title        = {LagrangeBench Datasets},
  month        = oct,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {0.0.1},
  url          = {https://zenodo.org/doi/10.5281/zenodo.10021925},
  doi          = {10.5281/zenodo.10021925},
}
```


### Publications
The following further publcations are based on the LagrangeBench codebase:

1. [Learning Lagrangian Fluid Mechanics with E(3)-Equivariant Graph Neural Networks (GSI 2023)](https://arxiv.org/abs/2305.15603), A. P. Toshev, G. Galletti, J. Brandstetter, S. Adami, N. A. Adams
