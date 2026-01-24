# -*- coding: utf-8 -*-
from collections import defaultdict
from typing import Optional, Callable

import torch
from torch.optim import Optimizer

class Lookahead(Optimizer):
    """
    Catalyst-style Lookahead wrapper (PyTorch 2.x compatible).

    Key points:
    - Do NOT call Optimizer.__init__ (avoids "empty parameter list")
    - Share param_groups/defaults with inner optimizer
    - Implement zero_grad so PyTorch 2.x won't touch base Optimizer.zero_grad
    """
    def __init__(self, optimizer: Optimizer, k: int = 5, alpha: float = 0.5):
        if not isinstance(optimizer, Optimizer):
            raise TypeError(f"optimizer must be torch.optim.Optimizer, got {type(optimizer)}")

        self.optimizer = optimizer
        self.k = int(k)
        self.alpha = float(alpha)

        # share groups/defaults with inner optimizer (as in Catalyst)
        self.param_groups = self.optimizer.param_groups
        self.defaults = self.optimizer.defaults
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state

        for group in self.param_groups:
            group.setdefault("counter", 0)

    def zero_grad(self, set_to_none: bool = False):
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def step(self, closure: Optional[Callable] = None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        return {
            "fast_state": fast_state_dict["state"],
            "slow_state": slow_state,
            "param_groups": fast_state_dict["param_groups"],
        }

    def load_state_dict(self, state_dict):
        # restore fast optimizer state
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

        # restore slow weights state
        id_map = {}
        for g in self.param_groups:
            for p in g["params"]:
                id_map[id(p)] = p

        self.state = defaultdict(dict)
        for k, v in state_dict["slow_state"].items():
            p = id_map.get(k, None)
            if p is not None:
                self.state[p] = v

    def add_param_group(self, param_group):
        param_group.setdefault("counter", 0)
        self.optimizer.add_param_group(param_group)