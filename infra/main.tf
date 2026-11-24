# Learning with Noisy Labels — single GCP training VM (g2-standard-4, 1x NVIDIA L4)
#
# All settings come from the environment: fill in ../.env (template: ../.env.example), then
#   set -a; source ../.env; set +a
# Terraform maps each TF_VAR_<name> to the variable <name> below. The defaults here are only
# fallbacks; project, region and zone are placeholders, and applying with a placeholder still in
# place fails with a clear validation error.

terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 8.0"
    }
  }
}

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------

variable "project_id" {
  description = "GCP project ID"
  type        = string
  default     = "<PROJECT_ID>"
  validation {
    condition     = !can(regex("^<.*>$", var.project_id))
    error_message = "Set project_id (still the placeholder)."
  }
}

variable "region" {
  description = "Region that offers L4 GPUs and has Spot L4 quota"
  type        = string
  default     = "<REGION>"
  validation {
    condition     = !can(regex("^<.*>$", var.region))
    error_message = "Set region (still the placeholder)."
  }
}

variable "zone" {
  description = "Zone inside the region that offers nvidia-l4"
  type        = string
  default     = "<ZONE>"
  validation {
    condition     = !can(regex("^<.*>$", var.zone))
    error_message = "Set zone (still the placeholder)."
  }
}

variable "credentials_file" {
  description = "Path to the Terraform service-account key (JSON)"
  type        = string
  default     = "~/.config/gcloud/tf-noisy-labels.json"
}

variable "instance_name" {
  type    = string
  default = "nl-trainer"
}

variable "machine_type" {
  type    = string
  default = "g2-standard-4"
}

variable "image_family" {
  description = "Deep Learning VM image family (list with the command in infra/README.md)"
  type        = string
  default     = "pytorch-2-9-cu129-ubuntu-2204-nvidia-580"
}

variable "disk_size_gb" {
  type    = number
  default = 50
}

variable "spot" {
  description = "true = Spot VM (cheap, preemptible, stops on preemption); false = on-demand"
  type        = bool
  default     = true
}

# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------

provider "google" {
  project     = var.project_id
  region      = var.region
  zone        = var.zone
  credentials = file(pathexpand(var.credentials_file))
}

# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------

# Deep Learning VM image: PyTorch + CUDA + NVIDIA driver preinstalled.
data "google_compute_image" "dlvm" {
  project = "deeplearning-platform-release"
  family  = var.image_family
}

resource "google_compute_instance" "trainer" {
  name         = var.instance_name
  machine_type = var.machine_type # g2-standard-4 attaches 1x NVIDIA L4 automatically
  zone         = var.zone

  labels = {
    project = "aai2-noisy-labels"
  }

  boot_disk {
    auto_delete = true
    initialize_params {
      image = data.google_compute_image.dlvm.self_link
      size  = var.disk_size_gb
      type  = "pd-balanced"
    }
  }

  network_interface {
    network = "default"
    access_config {} # ephemeral external IP for SSH / pip
  }

  scheduling {
    provisioning_model          = var.spot ? "SPOT" : "STANDARD"
    preemptible                 = var.spot
    automatic_restart           = var.spot ? false : true
    on_host_maintenance         = "TERMINATE" # required for GPU VMs
    instance_termination_action = var.spot ? "STOP" : null
  }

  # No service account: the VM never calls Google APIs, so it gets no credentials at all.

  metadata = {
    install-nvidia-driver = "True"
  }

  allow_stopping_for_update = true

  lifecycle {
    # `gcloud compute ssh` may add keys here; don't let Terraform fight over them.
    ignore_changes = [metadata["ssh-keys"]]
  }
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "instance_name" {
  value = google_compute_instance.trainer.name
}

output "zone" {
  value = google_compute_instance.trainer.zone
}

output "external_ip" {
  value = google_compute_instance.trainer.network_interface[0].access_config[0].nat_ip
}

output "ssh_command" {
  value = "gcloud compute ssh ${google_compute_instance.trainer.name} --zone ${google_compute_instance.trainer.zone} --project ${var.project_id}"
}

output "rsync_host" {
  description = "Host alias created by `gcloud compute config-ssh`"
  value       = "${google_compute_instance.trainer.name}.${google_compute_instance.trainer.zone}.${var.project_id}"
}
