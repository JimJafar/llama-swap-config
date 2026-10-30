# GPU power limits

`gpu-power-limits.sh` sets each GPU's power limit by UUID. Installed copies:

```sh
sudo install -m 755 gpu-power-limits.sh /usr/local/sbin/
sudo install -m 644 gpu-power-limits.service gpu-power-limits.timer /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-persistenced gpu-power-limits.service gpu-power-limits.timer
```

After editing the script here, re-run the first and last lines. Check with
`journalctl -u gpu-power-limits` or `nvidia-smi --query-gpu=name,power.limit --format=csv`.

Replaced LACT (`lactd`, `/etc/lact/config.yaml`), whose limits were keyed by PCI address and
silently fell back to stock when the EPYC upgrade moved the cards (2026-10-30). The reasons
for each value are in the script.
