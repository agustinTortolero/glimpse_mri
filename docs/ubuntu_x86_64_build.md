# Ubuntu x86_64 Build Guide

This guide describes the validated build flow for **Glimpse MRI** on Ubuntu x86_64.

## Scope

Target scripts are located in:

```bash
scripts/targets/ubuntu_x86_64/
```

The flow is intended for a development/build machine and produces a distributable tarball.

## 1) Install build prerequisites

From the repository root:

```bash
chmod +x scripts/targets/ubuntu_x86_64/*.sh
sudo ./scripts/targets/ubuntu_x86_64/prerequisites_build.sh
```

## 2) Build

```bash
./scripts/targets/ubuntu_x86_64/build.sh
```

This builds the project using the Ubuntu x86_64 target configuration.

## 3) Package

```bash
./scripts/targets/ubuntu_x86_64/pack.sh
```

The packaging step produces a release tarball that includes install/run helpers.

## 4) Runtime dependencies on a test machine (optional)

If deploying to a separate Ubuntu x86_64 machine, install runtime dependencies there:

```bash
sudo ./scripts/targets/ubuntu_x86_64/prerequisites_run.sh
```

## 5) Install and run from tarball

On the destination machine:

```bash
tar -xzf glimpse_mri_x86_64_Release_*.tar.gz
cd glimpse_mri_x86_64_Release_*
./install.sh
~/glimpse_mri/run.sh
```

## Notes

- Use this target for CPU/GPU-capable Ubuntu x86_64 Linux environments.
- Re-run `build.sh` after code updates; re-run `pack.sh` to publish a new artifact.
