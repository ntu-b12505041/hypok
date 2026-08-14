# K-MorphNet V2 experiment plan

K-MorphNet V2 is an ablation-controlled successor to the SE-ResNet baseline. It
does not claim that the clinical target has already been met; the locked test set
must remain untouched until a candidate is selected on validation data.

## Why multi-scale kernels?

At 250 Hz, the first-layer kernels span approximately 28, 60, and 124 ms:

- kernel 7 captures short, sharp local changes;
- kernel 15 captures intermediate wave morphology;
- kernel 31 captures broader waveform shape and context.

The three branches process the same lead in parallel and are fused with a 1x1
convolution. Deeper residual layers enlarge the effective receptive field. This
lets the encoder learn QRS and ST/T-related morphology without forcing one fixed
kernel width to represent all time scales.

## Controlled experiments

### V2-A: dual binary ablation

Configuration: `configs/experiments/mimic_v2a_dual_binary.yaml`

This retains the baseline SE-ResNet encoder but replaces the single three-class
head with separate HypoK-vs-rest and HyperK-vs-rest heads. It tests whether the
two clinically different abnormalities benefit from independent objectives.

### V2-B: K-MorphNet

Configuration: `configs/experiments/mimic_v2b_kmorphnet.yaml`

The model uses:

1. a shared per-lead multi-scale stem with kernels 7/15/31;
2. a shared SE-ResNet-34-style lead encoder (`[3, 4, 6, 3]` blocks);
3. temporal attention inside each lead;
4. a two-layer Transformer across the 12 leads;
5. independent HypoK and HyperK lead-attention experts;
6. dual binary, ordinal, and continuous-potassium objectives;
7. a small per-lead auxiliary loss to keep individual lead features informative.

Both experiments use the existing patient-disjoint split. Rotating NK
subsampling is training-only with ratio 2.5; validation and test retain natural
prevalence. Thresholds and head selection are fitted only on validation.

## Evidence behind the design

- DeepECG used a 34-layer SE-ResNet and separate low/high potassium detection
  tasks; its masking analysis supports preserving both QRS and T-wave regions:
  https://www.nature.com/articles/s41598-024-71562-5
- ECG12Net used shared per-lead processing and hierarchical lead attention:
  https://medinform.jmir.org/2020/3/e15931/
- A dynamic dyskalemia model used lead-wise blocks, lead attention, and ordinal
  potassium information:
  https://academic.oup.com/ehjdh/article/4/1/22/6840333
- Continuous/ordinal potassium estimation provides a useful auxiliary signal:
  https://www.nature.com/articles/s41598-024-65223-w

Published AUROC values are not directly comparable to this project's locked
three-class, patient-disjoint endpoint. The project target remains sensitivity
and specificity above 0.85 for every class on the untouched test set.

## Server commands

Run from the repository directory after activating the existing virtual
environment. Replace the path if the ECG directory differs.

```bash
cd ~/hypok/hypok
source .venv/bin/activate

git fetch origin
git switch agent/k-morphnet-v2
git pull --ff-only

sed -i 's#/path/to/mimic-iv-ecg/1.0#/home/bdm0162/hypok-data/mimic-iv-ecg/1.0#' \
  configs/experiments/mimic_v2a_dual_binary.yaml \
  configs/experiments/mimic_v2b_kmorphnet.yaml

python -m pip install -e . --no-deps
hypok-ecg validate-config \
  --config configs/experiments/mimic_v2a_dual_binary.yaml
hypok-ecg validate-config \
  --config configs/experiments/mimic_v2b_kmorphnet.yaml
```

Start with V2-A:

```bash
mkdir -p run_logs
tmux new -s v2a
hypok-ecg train \
  --config configs/experiments/mimic_v2a_dual_binary.yaml \
  2>&1 | tee "run_logs/08_train_v2a_$(date +%Y%m%d_%H%M%S).log"
```

Detach with `Ctrl-b`, then `d`. Reattach with `tmux attach -t v2a`.

After V2-A finishes, start V2-B in a new tmux session:

```bash
tmux new -s v2b
hypok-ecg train \
  --config configs/experiments/mimic_v2b_kmorphnet.yaml \
  2>&1 | tee "run_logs/09_train_v2b_$(date +%Y%m%d_%H%M%S).log"
```

Compare `metrics/calibration.json` and `logs/training_summary.json` under the two
new output directories. Do not run `evaluate` yet. Select the final candidate
using validation results first, then evaluate exactly once on the locked test
set.
