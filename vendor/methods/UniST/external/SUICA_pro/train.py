import argparse
from omegaconf import OmegaConf
from utils import pprint_config
from systems import train_embedder, train_inr, fit_griddata, predict_inr
from pathlib import Path
import os, h5py


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=['embedder', 'inr', 'grid'], required=True)
    parser.add_argument('--conf', type=str, required=True)
    args = parser.parse_args()
    configs = OmegaConf.load(args.conf)
    print(args.conf)
    pprint_config(configs)

    conf_path = Path(args.conf).resolve()
    conf_dir  = conf_path.parent
    repo_root = Path(__file__).resolve().parent  # Directory where train.py is located
    cwd       = Path.cwd()

    def resolve_rel(pth: str) -> Path:
        p = Path(pth)
        if p.is_absolute():
            return p
        # Try in order: working directory at commit time (-cwd) → repo root (train.py directory) → config file directory
        for base in (cwd, repo_root, conf_dir):
            cand = (base / p)
            if cand.exists():
                return cand.resolve()
        # If none exist, construct a candidate based on "repo root" and print it in error message
        return (repo_root / p).resolve()

    # Normalize paths
    configs.dataset.data_file = str(resolve_rel(str(configs.dataset.data_file)))
    configs.pipeline.optimization.logs = str(resolve_rel(str(configs.pipeline.optimization.logs)))

    print(f"[CONF] conf     = {conf_path}")
    print(f"[CONF] datafile = {configs.dataset.data_file}")
    print(f"[CONF] logs_dir = {configs.pipeline.optimization.logs}")

    p = configs.dataset.data_file
    if not os.path.exists(p):
        raise FileNotFoundError(
            f"data_file does not exist: {p}\n"
            f"Please check -cwd or change data_file in YAML to an absolute path."
        )

    # Lightweight pre-check: whether .h5ad.gz was mistakenly passed
    with open(p, "rb") as f:
        if f.read(2) == b"\x1f\x8b":
            raise RuntimeError(f"{p} appears to be a gzip archive (.h5ad.gz), please gunzip it to .h5ad first")

    if not h5py.is_hdf5(p):
        raise RuntimeError(f"{p} is not an HDF5 (.h5ad) file: please check if the path/file is corrupted or is a zarr directory")

    if args.mode == "embedder":
        train_embedder(configs)
    elif args.mode == "inr":
        train_inr(configs)
    elif args.mode == "grid":
        fit_griddata(configs)

    pprint_config(configs)