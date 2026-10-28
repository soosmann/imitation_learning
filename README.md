# Push-T Imitation Learning

A PyTorch implementation of behavioral cloning with action chunking for the
Push-T task, developed for Berkeley deep reinforcement learning homework 1.
The project compares two state-conditioned MLP policies:

- **MSE policy:** predicts a chunk of expert actions directly using mean squared error.
- **Flow matching policy:** learns a velocity field along a linear path from
  Gaussian noise to expert action chunks, then generates actions using Euler integration.

Both policies use normalized states and actions. Action chunks stay within
individual demonstration episodes, and the policy executes the full chunk before
requesting another one.

## Example rollouts

Evaluation episode 0 after 400 training epochs (training seed 42):

- [Flow matching policy](assets/rollouts/flow_epoch400_ep0.mp4)
- [MSE policy](assets/rollouts/mse_epoch400_ep0.mp4)

Only these two selected videos are included in the repository; generated training
and inference videos are excluded by `.gitignore`.

The corresponding experiment directories are included with their configurations,
normalization statistics, best-model weights, metrics, and training plots:

- `exp/seed_42_20261028_220725/` (flow matching)
- `exp/seed_42_20261028_220006/` (MSE)

Their `videos/` subdirectories remain excluded.

## Setup

Use Python 3.11 (the version in `.python-version`) and `uv`. Run all commands from
the project root.

```bash
uv sync --locked
uv run src/train.py --help
```

The dependencies include the packages used for training, evaluation, and MP4
recording. Experiments are logged locally; no experiment tracking account is required.

## Training

Train the flow matching policy with the default configuration:

```bash
uv run src/train.py --policy-type flow --exp-name flow_baseline
```

Train the MSE policy with the same architecture and action chunk size:

```bash
uv run src/train.py --policy-type mse --exp-name mse_baseline
```

On the first run, the training script downloads and extracts the Push-T
demonstration dataset into `data/`. Existing extracted data is reused.
The dataset download URL is defined as `PUSHT_URL` in `src/data.py`.

The script automatically selects Apple MPS, CUDA, or CPU, in that order.
For a short training run without evaluation or video recording:

```bash
uv run src/train.py --num-epochs 1 --eval-interval 10 --num-video-episodes 0
```

This still loads the full dataset and trains for one epoch.

### Configuration

Run `uv run src/train.py --help` for the full list of options.

| Option | Default | Purpose |
| --- | --- | --- |
| `--policy-type` | `flow` | Select `mse` or `flow`. |
| `--chunk-size` | `8` | Actions generated and executed per policy call. |
| `--hidden-dims` | `256 256 256` | Width of each MLP hidden layer. |
| `--batch-size` | `128` | Demonstration chunks per training batch. |
| `--lr` | `0.0003` | Adam learning rate. |
| `--weight-decay` | `0.0` | Adam weight decay. |
| `--num-epochs` | `400` | Number of training epochs. |
| `--eval-interval` | `10` | Evaluate every this many epochs. |
| `--flow-num-steps` | `10` | Euler integration steps for flow sampling; ignored by MSE. |
| `--num-video-episodes` | `3` | Evaluation episodes to record when video recording is due. |
| `--video-every-n-steps` | `50` | Epoch interval for recording; must coincide with evaluation. |
| `--video-size` | `256 256` | Video width and height. |
| `--seed` | `42` | NumPy and PyTorch random seed. |
| `--data-dir` | `data` | Dataset download and extraction directory. |
| `--exp-name` | None | Optional suffix for the experiment directory. |

Set `--num-video-episodes 0` or `--video-every-n-steps 0` to disable videos.

## Evaluation and outputs

Evaluation runs 100 episodes with fixed environment seeds (`0` through `99`).
The reported reward is the **mean of the maximum reward reached in each episode**,
not the sum of rewards. Actions are denormalized and clipped to the environment's
allowed range before execution.

Each training run creates a directory under `exp/`:

```text
exp/seed_<seed>_<timestamp>[_<exp_name>]/
├── config.json             # Full training configuration, including CLI overrides
├── normalizer.npz          # State and action means and standard deviations
├── log.csv                 # Training loss and evaluation metrics, one row per epoch
├── training_curves.png     # Refreshed after each evaluation
├── best_model.pt           # Policy state_dict with the best evaluation reward
├── best_reward.txt         # Reward corresponding to the best saved weights
└── videos/                 # Created when video recording is due
    └── eval_step_<epoch>_ep<episode>.mp4
```

The configuration and normalization statistics are saved before training starts.
Evaluation artifacts appear after evaluation runs. Keep `best_model.pt`,
`config.json`, and `normalizer.npz` together to reconstruct a trained policy:
the JSON records the policy type, hidden layer sizes, action chunk size, and
sampling settings, while the NPZ stores `state_mean`, `state_std`, `action_mean`,
and `action_std`. State and action dimensions can be read from these arrays.

`best_model.pt` retains the standard PyTorch `state_dict` format. Optimizer state
is not saved, and there is no training resume command.

To regenerate plots from an existing metrics CSV:

```bash
uv run src/plot_training.py exp/<experiment_directory>/log.csv
```

Training losses for MSE and flow matching measure different objectives; use the
shared evaluation reward to compare task performance.

## Inference

Load a saved policy and run one episode:

```bash
uv run src/inference.py --exp-dir exp/<experiment_directory>
```

The script restores the architecture, sampling settings, and normalization
statistics from the experiment directory. It executes action chunks in the same
way as evaluation and reports each episode's maximum reward and their mean.
The directory must contain `best_model.pt`, `config.json`, and `normalizer.npz`.

Run multiple episodes and save videos:

```bash
uv run src/inference.py --exp-dir exp/<experiment_directory> \
    --num-episodes 3 --seed 0 --video-dir videos/inference
```

Use `--render` to display a live rollout instead of recording videos. Inference
automatically selects MPS, CUDA, or CPU; use `--device cpu` to select CPU explicitly.
Run `uv run src/inference.py --help` for all options.

## Project structure

```text
src/
├── data.py             # Dataset download, normalization, and episode-safe chunks
├── model.py            # MSE and flow matching policies
├── train.py            # Configuration, command-line interface, and training loop
├── evaluation.py       # Rollouts, CSV logging, videos, and best-weight saving
├── inference.py        # Load a saved policy and run episodes or record videos
└── plot_training.py    # Training loss and evaluation reward plots
```

`data/`, virtual environments, generated videos, and new experiment directories
are excluded from Git by `.gitignore`. The two example experiments listed above
are explicitly included. Commit `pyproject.toml`, `uv.lock`, and `.python-version`
alongside the source to preserve the dependency configuration.
