""" Module for handling file I/O operations """

from pathlib import Path
import yaml
import pandas as pd

def load_yaml(path: Path) -> dict:
    """Load a YAML file and return its contents as a dictionary."""
    with open(path, 'r') as f:
        return yaml.safe_load(f)
    
def save_parquet(df: pd.DataFrame, path: Path):
    """Save a DataFrame to a parquet file, creating the directory if it doesn't exist."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def load_parquet(path: Path) -> pd.DataFrame:
    """Load a parquet file and return it as a DataFrame."""
    return pd.read_parquet(path)

def ensure_dir(path: Path):
    """Ensure that a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)