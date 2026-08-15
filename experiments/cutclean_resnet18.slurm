#!/bin/bash
#
# SLURM job template for a single CutClean run.
# Submitted by experiments/run_resnet18_paper.sh with USE_SLURM=1, which passes
# the driver path and all of its arguments as "$@" and sets --job-name/--output.
#
# It can also be submitted by hand:
#   sbatch experiments/cutclean_resnet18.slurm src/run_cutclean.py \
#          --dataset celeba-Blond_Hair-Male --datapath /path/to/data/ \
#          --projectName my_run --gamma_list 1000 --sparsity_list 0.6 --t_ph 0.65
#
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8

# No `set -u`: conda's activation hooks reference unbound variables
# (MKL_INTERFACE_LAYER) and would abort the job before training starts.
set -eo pipefail

CONDA_ENV="${CONDA_ENV:-cutclean}"

echo "=== CutClean job ${SLURM_JOB_NAME:-manual} ==="
echo "JOB ID    : ${SLURM_JOB_ID:-n/a}"
echo "HOSTNAME  : $(hostname)"
echo "START     : $(date)"

if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    conda activate "$CONDA_ENV"
else
    echo "WARNING: conda.sh not found; using the interpreter already on PATH."
fi

echo "PWD       : $(pwd)"
echo "PYTHON    : $(which python)  ($(python -V 2>&1))"
echo "GPU       : ${CUDA_VISIBLE_DEVICES:-<unset>}"
command -v nvidia-smi >/dev/null && nvidia-smi || echo "nvidia-smi not available"

python "$@"
EXIT_CODE=$?

echo "EXIT CODE : $EXIT_CODE"
echo "END       : $(date)"
exit $EXIT_CODE
