# run.py
import sys
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent

# Absolute paths to script and config
script_module = "/Users/erikborn/Documents/Python/SalaryAnalysis/scripts/run_pipeline.py"
config_path   = ROOT / "config.yaml"

# Fake the CLI args
sys.argv = [script_module, "--config", str(config_path)]

# Run as if `python -m scripts.run_pipeline --config config.yaml`
runpy.run_module(script_module, run_name="__main__")