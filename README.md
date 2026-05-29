# Iterative Robust Satisficing

Code and experiments for "Iterative Robust Satisficing: Minimizing Performance Degradation Under Distribution Shift" (ICML 2026).

## Repository Layout

```text
code/
|-- cifar10lt_experiment.py      # CIFAR-10-LT long-tailed classification experiment
|-- tabular.ipynb                # Tabular experiment
|-- waterbirds.ipynb             # Waterbirds spurious-correlation experiment
|-- terraincognito.ipynb         # TerraIncognita domain-generalization experiment
`-- methods/
    |-- irs.py                   # Iterative Robust Satisficing
    |-- klrs.py                  # KL-RS baseline
    |-- erm.py                   # ERM with SGD
    |-- erm_adam.py              # ERM with Adam
    |-- sam.py                   # Sharpness-Aware Minimization
    |-- groupdro.py              # GroupDRO
    |-- chi2_dro.py              # Chi-squared DRO
    |-- cvar_dro.py              # CVaR-DRO
    |-- irm.py                   # IRMv1
    |-- vrex.py                  # V-REx
    `-- mmrex.py                 # MM-REx
```

## Environment

The code is written in Python/PyTorch. A typical setup is:

```bash
pip install torch torchvision numpy pandas matplotlib pillow tqdm timm seaborn
```

CUDA is recommended for the vision experiments.

## CIFAR-10-LT

Run:

```bash
cd code
python cifar10lt_experiment.py
```

This experiment runs CIFAR-10-LT long-tailed classification. CIFAR-10 is downloaded automatically through `torchvision` into `./data`.

Main settings are defined in `CIFAR10LTConfig`, including the imbalance factor, seeds, methods, learning rates, number of epochs, and output directory.

Outputs include per-epoch CSVs, class-wise accuracy CSVs, tail accuracy printed at the end of each method, `config.json`, and `comparison.png`.

## Tabular

Run the notebook:

```text
tabular.ipynb
```

Open the notebook and run the cells in order. The notebook contains the tabular experiment, including default hyperparameters, optional hyperparameter search, and final multi-seed training.

## Waterbirds

Run the notebook:

```text
waterbirds.ipynb
```

Open `waterbirds.ipynb` and run the cells in order. The notebook contains the Waterbirds spurious-correlation experiment.

The notebook expects/prepares Waterbirds locally in the code directory. The expected prepared structure is either:

```text
data/waterbirds/
|-- metadata.csv
`-- images/
```

or:

```text
data/waterbird_complete95_forest2water2/
|-- metadata.csv
`-- images/
```

The notebook prints validation/test metrics, group accuracies, and runtime summaries for the enabled methods.

## TerraIncognita

Run the notebook:

```text
terraincognito.ipynb
```

The notebook uses DomainBed TerraIncognita, fine-tunes an ImageNet-1k-pretrained DeiT-S/16 backbone for 10 epochs, and evaluates on a held-out location.

The notebook saves checkpoints and optional test outputs for significance testing.
