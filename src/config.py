# src/config.py
import toml
from pathlib import Path

def load_config():
    path = Path(__file__).resolve().parents[1] / "config.toml"
    return toml.load(path)