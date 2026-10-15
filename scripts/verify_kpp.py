"""Verify KPP imports and, by default, a small licensed SA + RBM training loop.

Run with the target KPP environment's Python. Use --imports-only before the
user's Kaiwu license is configured; that mode does not verify SDK solving.
The default mode calls kaiwu.license.init when KAIWU_USER_ID and KAIWU_SDK_CODE
are both set; without a license it prints KPP_LICENSE_MISSING and exits 2
instead of waiting for console input.
"""
from __future__ import annotations

import argparse
from importlib.metadata import version
import os
import sys

import numpy as np
import torch
import kaiwu.license
from kaiwu.classical import SimulatedAnnealingOptimizer
from kaiwu.torch_plugin import (
    BoltzmannMachine,
    FeatureSelectionWrapper,
    QDiffusion,
    QVAE,
    RestrictedBoltzmannMachine,
    UnsupervisedDBN,
)

CORE_MODELS = (
    RestrictedBoltzmannMachine, BoltzmannMachine, UnsupervisedDBN, QVAE, QDiffusion,
    FeatureSelectionWrapper,
)
LICENSE_MISSING = "KPP_LICENSE_MISSING: 未找到 Kaiwu license，设置 KAIWU_USER_ID/KAIWU_SDK_CODE 后重试"


def main() -> None:
    """Check public model imports and optionally run the licensed RBM check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--imports-only", action="store_true")
    args = parser.parse_args()

    print("python:", sys.executable)
    for name in ("kaiwu-torch-plugin", "kaiwu", "torch", "numpy"):
        print(f"{name}=={version(name)}")
    print("core_models:", ", ".join(model.__name__ for model in CORE_MODELS))
    print("KPP_IMPORT_OK", flush=True)
    if args.imports_only:
        return

    user_id, sdk_code = os.environ.get("KAIWU_USER_ID"), os.environ.get("KAIWU_SDK_CODE")
    if user_id and sdk_code:
        kaiwu.license.init(user_id=user_id, sdk_code=sdk_code)
    try:
        train_check()
    except EOFError:
        print(LICENSE_MISSING)
        sys.exit(2)
    except ValueError as error:
        if "EOF" not in str(error):
            raise
        print(LICENSE_MISSING)
        sys.exit(2)


def train_check() -> None:
    """Run three RBM steps with SDK SA sampling and check shapes, gradients, updates."""
    torch.manual_seed(7)
    np.random.seed(7)
    rbm = RestrictedBoltzmannMachine(num_visible=4, num_hidden=3).to("cpu")
    optimizer = torch.optim.SGD(rbm.parameters(), lr=0.01)
    sampler = SimulatedAnnealingOptimizer(
        initial_temperature=10, cutoff_temperature=0.1,
        alpha=0.8, iterations_per_t=4, size_limit=4, rand_seed=7,
    )
    before = [parameter.detach().clone() for parameter in rbm.parameters()]
    width = rbm.num_visible + rbm.num_hidden
    for step in range(3):
        optimizer.zero_grad()
        batch = torch.randint(0, 2, (8, rbm.num_visible), dtype=torch.float32)
        positive = rbm.get_hidden(batch, bernoulli=True)
        negative = rbm.sample(sampler)
        if positive.shape != (8, width):
            raise RuntimeError(f"Unexpected positive shape: {tuple(positive.shape)}")
        if negative.ndim != 2 or negative.shape[0] == 0 or negative.shape[1] != width:
            raise RuntimeError(f"Unexpected negative shape: {tuple(negative.shape)}")
        if not bool(((negative == 0) | (negative == 1)).all()):
            raise RuntimeError("SDK samples were not converted to binary joint states")
        loss = rbm.objective(positive, negative)
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            raise RuntimeError("RBM objective is not a finite scalar")
        loss.backward()
        grads = [p.grad for p in rbm.parameters() if p.grad is not None]
        if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
            raise RuntimeError("Missing or non-finite gradients")
        optimizer.step()
        print(f"step={step + 1} loss={loss.item():.6f} sample_shape={tuple(negative.shape)}")
    if not any(not torch.equal(old, new.detach()) for old, new in zip(before, rbm.parameters())):
        raise RuntimeError("Optimizer did not update any model parameter")
    print("KPP_TRAIN_OK: SDK SA sampling + binary states + gradients + parameter update")


if __name__ == "__main__":
    main()
