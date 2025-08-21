#!/bin/bash
export PYTHONPATH=$(pwd)
pytest --import-mode=importlib --cov=causalllm tests/
