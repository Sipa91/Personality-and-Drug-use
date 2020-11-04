SHELL := /bin/bash

.PHONY: setup-conda
setup-conda:
	conda env create -n ds-druguse -f druguse.yml

.PHONY: setup-pyenv
setup-pyenv:
	pyenv local 3.8.5
	python -m venv .venv
	source .venv/bin/activate && pip install -r requirements.txt