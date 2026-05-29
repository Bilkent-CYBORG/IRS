# Iterative Robust Satisficing
Code and experiments of "Iterative Robust Satisficing: Minimizing Performance Degradation Under Distribution Shift" (ICML 2026).

## Repository Layout

```text
code/
|-- cifar10lt_experiment.py      # CIFAR-10-LT long-tailed classification experiment
|-- waterbirds_experiment.py     # Waterbirds spurious-correlation experiment
|-- terraincognito.ipynb         # Additional TerraIncognita domain-generalization experiment
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

The main experiment files define a configuration dataclass near the top of the file and instantiate it at the bottom as `CONFIG = ...Config()`. To change seeds, methods, learning rates, imbalance factors, or output folders, edit this config before running.

## CIFAR-10-LT

Run:

```bash
cd code
python cifar10lt_experiment.py
```

By default, this runs CIFAR-10-LT with imbalance factor `100`, WideResNet-28-10, seed `42`, and the methods listed in `CIFAR10LTConfig.methods`. CIFAR-10 is downloaded automatically through `torchvision` into `./data`.


Outputs include per-epoch CSVs, class-wise accuracy CSVs, tail accuracy printed at the end of each method, `config.json`, and `comparison.png`.

## Waterbirds

Run:

```bash
cd code
python waterbirds_experiment.py
```

By default, the script expects Waterbirds under `./data/waterbirds`. If `download=True`, it attempts to download and prepare the dataset automatically. The expected prepared structure is:

```text
data/waterbirds/
|-- metadata.csv
`-- images/
```

Outputs include per-epoch CSVs, per-group test metrics, `summary_per_seed.csv`, `summary_aggregated.csv`, `group_accuracy_summary.csv`, `config.json`, and `comparison.png`.

## TerraIncognita

The TerraIncognita experiment is provided as a notebook:

```text
terraincognito.ipynb
```

It is intended to be run in Colab or another notebook environment. The notebook uses DomainBed TerraIncognita, fine-tunes an ImageNet-1k-pretrained DeiT-S/16 backbone for 10 epochs, and evaluates on a held-out location. The default held-out domain is `location_100`.

Important notebook settings:

- `MANUAL_TERRA_ROOT`: set this if the TerraIncognita folder is already available locally.
- `DOWNLOAD_WITH_DOMAINBED_IF_MISSING`: attempts to fetch TerraIncognita through DomainBed if needed.
- `TEST_DOMAIN_SELECTOR`: held-out location, default `"location_100"`.
- `METHOD_ENABLED`: enables/disables ERM, SAM, CVaR-DRO, and IRS-Instance.
- `SHARED_HPARAMS` and `METHOD_HPARAMS`: training and method-specific hyperparameters.

The notebook saves checkpoints and optional test outputs for significance testing.

## Method Keys

The experiments support the following method keys:

```text
irs, klrs, erm, erm_adam, erm_sgd, sam, vrex, mmrex, irm, groupdro, chi2_dro, cvar_dro
```

Not every key is used in every experiment. If an unknown key is listed in `methods`, the script prints the available keys for that experiment.

