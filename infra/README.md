# Infrastructure

`main.tf` creates **one** GCP virtual machine for training and nothing else:

| Setting | Value | Why |
|---|---|---|
| Machine | `g2-standard-4`: 1× NVIDIA L4 (24 GB), 4 vCPU, 16 GB RAM | the dataset lives on the GPU, so 4 CPUs are enough |
| Pricing | Spot (`SPOT=true` in `.env`) | about a third of the on-demand price; may be stopped by Google at any time |
| On preemption | the VM is **stopped**, not deleted | disk and results survive; start it again and re-run the same command |
| Disk | 100 GB balanced persistent disk | the smallest size the image allows; CIFAR-10 is loaded into memory once, so a faster SSD would not help |
| Image | Deep Learning VM (`pytorch-2-9-cu129-ubuntu-2204-nvidia-580`) | NVIDIA driver preinstalled |
| Service account | none | the VM never calls Google APIs, so it holds no Google credentials |

All values come from `.env` (template: `.env.example`) via `TF_VAR_*` variables, so the commands
need no flags:

```bash
set -a; source .env; set +a
terraform -chdir=infra init
terraform -chdir=infra apply      # create
terraform -chdir=infra destroy    # delete everything
```

The full procedure (one-time GCP setup, SSH, pausing and resuming, tear-down) is in
[../RUNBOOK.md](../RUNBOOK.md), steps 1–3, 11 and 14.
