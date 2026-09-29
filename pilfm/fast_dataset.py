"""Zero-copy subclass of the official CylinderHFDataset.

The official __getitem__ (fluid_hf_dataset.py:261-338) materializes a whole trajectory row
(`self.trajectories[traj_idx]`, ~0.5 GB per field for numerical) to slice 40 frames out of it.
Here the same slice is taken directly from the memory-mapped Arrow buffer. Everything else --
channel stacking, the random.random() mask draw, the noise draws and their order -- is copied line
for line, so outputs are bitwise identical given the same RNG state (checked in fm/tests).
"""
import random

import numpy as np
import torch

from realpdebench.data.fluid_hf_dataset import CylinderHFDataset
from realpdebench.data.dataset import apply_gaussian_blur


class FastCylinderHFDataset(CylinderHFDataset):
    def _table(self):
        # Arrow table behind the datasets.Dataset; opened lazily so it is re-mapped inside each worker
        if getattr(self, "_tab", None) is None:
            self._tab = self.trajectories.data.table
        return self._tab

    def _field(self, name, traj_idx, shape):
        buf = self._table().column(name)[traj_idx].as_buffer()
        return np.frombuffer(buf, dtype=np.float32).reshape(shape)

    def __getitem__(self, idx):
        entry = self._indices[idx]
        sim_id, time_id = entry["sim_id"], entry["time_id"]
        traj_idx = self._sim_id_to_idx[sim_id]
        tab = self._table()
        full_shape = tuple(int(tab.column(k)[traj_idx].as_py()) for k in ("shape_t", "shape_h", "shape_w"))
        sub_s = self.sub_s
        sl = (slice(time_id, time_id + self.horizon), slice(None, None, sub_s), slice(None, None, sub_s))
        u = self._field("u", traj_idx, full_shape)[sl]
        v = self._field("v", traj_idx, full_shape)[sl]
        if self.dataset_type == "real":
            p = np.zeros_like(u)
        else:
            if random.random() < self.mask_prob:
                p = np.zeros_like(u)
            else:
                p = self._field("p", traj_idx, full_shape)[sl]
        data = np.stack([u, v, p], axis=-1)
        input_data = torch.tensor(data[:self.in_step], dtype=torch.float32)
        output_data = torch.tensor(data[self.in_step:], dtype=torch.float32)
        if self.noise_scale > 0 and self.dataset_type == "numerical":
            if self.noise_type == "gaussian":
                input_data = input_data + input_data * torch.randn_like(input_data) * self.noise_scale
                output_data = output_data + output_data * torch.randn_like(output_data) * self.noise_scale
            elif self.noise_type == "poisson":
                input_data = input_data + torch.poisson(input_data) * self.noise_scale
                output_data = output_data + torch.poisson(output_data) * self.noise_scale
            elif self.noise_type == "optical":
                input_data = apply_gaussian_blur(input_data, self.optical_kernel_size, self.optical_sigma)
                output_data = apply_gaussian_blur(output_data, self.optical_kernel_size, self.optical_sigma)
            else:
                raise ValueError(f"Invalid noise type: {self.noise_type}")
        return input_data, output_data

    def __getstate__(self):
        d = self.__dict__.copy()
        d["_tab"] = None
        return d
