#!/usr/bin/env python3
"""Host<->GPU copy bandwidth per GPU over PCIe (pinned host memory, 256 MiB, best of 20).

Run with a torch-enabled Python, e.g. ~/comfyui/ComfyUI/.venv/bin/python bench/pcie_bw.py
Needs ~256 MiB free VRAM on each card. Also prints the negotiated PCIe link, read
from sysfs WHILE a copy is running (idle cards drop to 2.5 GT/s).
"""
import os
import pathlib
import threading
import time

# torch numbers GPUs fastest-first by default; match nvidia-smi / PCI order instead
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
import torch  # noqa: E402

MIB = 256
REPS = 20


def link(bus_id):
    d = pathlib.Path("/sys/bus/pci/devices") / bus_id.lower()
    up = d.resolve().parent
    rd = lambda p, f: (p / f).read_text().strip() if (p / f).exists() else "?"
    return (f"{rd(d, 'current_link_speed')} x{rd(d, 'current_link_width')} "
            f"(card max {rd(d, 'max_link_speed')} x{rd(d, 'max_link_width')}, "
            f"slot max {rd(up, 'max_link_speed')} x{rd(up, 'max_link_width')})")


def bus_ids():
    import subprocess
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id,name", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout
    res = {}
    for line in out.strip().splitlines():
        i, bus, name = [x.strip() for x in line.split(",", 2)]
        res[int(i)] = ("0000:" + bus[-7:], name)
    return res


def measure(dev, src, dst):
    torch.cuda.synchronize(dev)
    best = 1e9
    for _ in range(REPS):
        t = time.perf_counter()
        dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(dev)
        best = min(best, time.perf_counter() - t)
    return MIB * 2**20 / best / 1e9


def main():
    ids = bus_ids()
    n = MIB * 2**20
    host = torch.empty(n, dtype=torch.uint8).pin_memory()
    for i in range(torch.cuda.device_count()):
        dev = torch.device("cuda", i)
        gpu = torch.empty(n, dtype=torch.uint8, device=dev)
        bus, name = ids.get(i, ("?", torch.cuda.get_device_name(i)))
        seen = {}
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                seen[link(bus)] = 1
                time.sleep(0.02)

        th = threading.Thread(target=sample)
        th.start()
        h2d = measure(dev, host, gpu)
        d2h = measure(dev, gpu, host)
        stop.set()
        th.join()
        lk = max(seen, key=lambda s: s.split()[0]) if seen else link(bus)
        print(f"GPU{i} {name:26s} {bus}  H2D {h2d:5.1f} GB/s  D2H {d2h:5.1f} GB/s  link {lk}")
        del gpu
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
