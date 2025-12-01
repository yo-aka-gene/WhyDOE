import logging
import os
import shutil
import subprocess
import sys

import anndata as ad
import numpy as np
import pandas as pd
import rpy2.robjects as ro
import rpy2.robjects.vectors as ro_vectors
from rpy2.robjects.packages import importr, isinstalled
from rpy2.robjects import numpy2ri, pandas2ri
import rpy2.rinterface_lib.callbacks


utils = importr('utils')
base = importr('base')
packages = ["BiocManager"]
repos = "https://cloud.r-project.org/"
rscript = f"{os.path.dirname(__file__)}/splat.R"
func_name = "generate_clusters"
R_FUNC = None


rpy2.rinterface_lib.callbacks.consolewrite_print = lambda x: None
rpy2.rinterface_lib.callbacks.consolewrite_warnerror = lambda x: None
pandas2ri.activate()
numpy2ri.activate()


sys_deps = [
    "libcairo2-dev",
    "libxt-dev",
    "libgirepository1.0-dev",
    "libcurl4-openssl-dev",
    "libssl-dev",
    "libxml2-dev"
]


def _install_system_dependencies(deps: list = sys_deps):
    if shutil.which("apt-get") is None:
        return

    missing_packages = []

    for pkg in deps:
        result = subprocess.run(
            ["dpkg", "-s", pkg],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        if result.returncode != 0:
            missing_packages.append(pkg)

    if missing_packages:
        print(f"[System] Installing missing dependencies: {', '.join(missing_packages)}")
        try:
            subprocess.run(["sudo", "apt-get", "update"], check=True)
            subprocess.run(
                ["sudo", "apt-get", "install", "-y"] + missing_packages,
                check=True
            )
            print("[System] Dependencies installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"[Warning] Failed to install system dependencies: {e}")
            print("You may need to install them manually or check sudo permissions.")
        except FileNotFoundError:
            try:
                subprocess.run(["apt-get", "update"], check=True)
                subprocess.run(["apt-get", "install", "-y"] + missing_packages, check=True)
            except Exception:
                print("[Warning] Could not install packages. Please install manually.")


_install_system_dependencies()


for pkg in packages:
    if not isinstalled(pkg): 
        utils.install_packages(pkg, repos=repos)
    base.suppressPackageStartupMessages(
        base.library(pkg, character_only=ro_vectors.BoolVector([True]))
    )

bioc_manager = importr('BiocManager')

bio_packages = ["splatter", "scater"]

for pkg in bio_packages:
    if not isinstalled(pkg): 
        bioc_manager.install(
            pkg,
            update=False,
            ask=False
        )
    base.suppressPackageStartupMessages(
        base.library(pkg, character_only=ro_vectors.BoolVector([True]))
    )

with open(rscript) as f:
    ro.r(f.read())


def _initialize_r_func():
    global R_FUNC
    if R_FUNC is None:
        R_FUNC = ro.r[func_name]


def generate_clusters(
    n_genes: int,
    n_cells: int,
    group_prob: np.ndarray,
    de_prob: float,
    dropout_mid: float,
    random_state: int = 0
) -> ad.AnnData:
    _initialize_r_func()
    
    if group_prob.sum() != 1:
        group_prob /= group_prob.sum()
    
    simulated = R_FUNC(
        n_genes=n_genes,
        n_cells=n_cells,
        group_prob=group_prob,
        de_prob=de_prob,
        dropout_mid=dropout_mid,
        random_state=random_state
    )
    
    count = simulated.rx2("counts").T
    meta = simulated.rx2("meta")
    
    adata = ad.AnnData(X=count)
    # adata.obs = meta
    # adata.obs["Group"] = adata.obs["Group"].astype["category"]
    
    return adata