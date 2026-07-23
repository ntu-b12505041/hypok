PYTHON ?= python
CONFIG ?= configs/mimic.yaml

.PHONY: install test demo cohort train evaluate download-founder train-founder evaluate-founder compare

install:
	$(PYTHON) -m pip install -e ".[dev,foundation]"

test:
	$(PYTHON) -m pytest

demo:
	$(PYTHON) scripts/make_synthetic_demo.py --output-dir outputs/synthetic_demo

cohort:
	hypok-ecg build-cohort --config $(CONFIG)

train:
	hypok-ecg train --config $(CONFIG)

evaluate:
	hypok-ecg evaluate --config $(CONFIG)

download-founder:
	$(PYTHON) scripts/download_ecgfounder.py

train-founder:
	hypok-ecg train --config configs/ecgfounder_finetune.yaml

evaluate-founder:
	hypok-ecg evaluate --config configs/ecgfounder_finetune.yaml

compare:
	hypok-ecg compare --baseline-config configs/mimic.yaml --finetune-config configs/ecgfounder_finetune.yaml
