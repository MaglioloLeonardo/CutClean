# CutClean: Neural Network Pruning for Privacy-Preserving Inference

Reference implementation of the method described in:

> Leonardo Magliolo, Vito Paolo Pastore, Giuseppe Valenzise, Enzo Tartaglione.
> **CutClean: Neural Network Pruning for Privacy-Preserving Inference.**
> *Pattern Recognition* — Proceedings of the 28th International Conference on Pattern
> Recognition (ICPR 2026), Lyon, France, August 17–22, 2026.
> Lecture Notes in Computer Science, Springer Nature Switzerland, pp. 450–465.
> [doi:10.1007/978-3-032-31452-9_30](https://doi.org/10.1007/978-3-032-31452-9_30)
[Arxiv preprint](http://arxiv.org/abs/2608.13773)

Leonardo Magliolo and Enzo Tartaglione are with LTCI, Télécom Paris, Institut
Polytechnique de Paris, France; Vito Paolo Pastore is with MaLGa-DIBRIS, University of
Genoa and Istituto Italiano di Tecnologia, Italy; Giuseppe Valenzise is with Université
Paris-Saclay, CNRS, CentraleSupélec, L2S, France.
Correspondence: `{leonardo.magliolo, enzo.tartaglione}@telecom-paris.fr`

## Abstract

Neural networks are increasingly deployed in high-stakes applications with growing privacy
leakage concerns. We show that this privacy leakage can occur even in the absence of
representation imbalances that lead to traditional dataset biases. This poses significant
privacy risks when deploying models that process sensitive attributes. In this context, we
propose CutClean, a privacy-aware pruning method that allows to reduce privacy information
flow through the network, while increasing its sparsity. Our approach employs auxiliary
linear privacy heads placed at each network's block to quantify information leakage, and
further applies increasing levels of sparsity to remove the private attribute leakage,
measured in terms of the accuracy of the privacy head attached to the last block.
Experiments on synthetic and real-world datasets demonstrate that our approach effectively
minimizes private information flow while achieving high sparsity rates and preserving
classification target accuracy.

## Method

A network `f = g_c ∘ b_B ∘ ... ∘ b_1` is instrumented with one auxiliary linear *privacy
head* `g_p^i` per block, each attached to the block output `h_i` through a forward hook and
trained to predict the private attribute `z`. The accuracy of the head attached to the last
block is used as the measure of private information flow.

Training minimises the objective

```
J = L_y(ŷ, y) + Σ_i γ_i · I_z(ẑ_i, z)
```

where `I_z` is a differentiable mutual-information proxy computed from the empirical joint
distribution of the head predictions and the private labels over the mini-batch. Contrary
to maximising the privacy cross-entropy, minimising `I_z` avoids the degenerate solution in
which the head systematically misclassifies the private attribute while the backbone still
encodes it.

The pipeline has three stages:

1. **Online batch MI-aware training.** For every mini-batch, the privacy heads are updated
   first with the backbone frozen, the forward pass is then recomputed, and the backbone and
   task head are updated on `J`. Alternating within the batch keeps the mutual-information
   estimate aligned with heads that are optimal for the current representation.
2. **γ selection.** The procedure above is run for each candidate `γ`, applied uniformly to
   all heads. The backbone is then frozen, the privacy heads are retrained from scratch, and
   the value yielding the lowest validation accuracy of the last head is retained as `γ*`.
3. **Privacy-aware pruning.** Starting from the backbone obtained with `γ*`, a grid of global
   sparsity levels `s ∈ [0, 1]` is explored. Each candidate is pruned by removing the output
   channels with the lowest normalised L1 norm, fine-tuned for a small number of epochs with
   the same alternating scheme, and evaluated after retraining the privacy heads. A sparsity
   level is *admissible* when the last privacy head stays below a threshold `P_threshold`
   defined by the number of private classes. Among the admissible models, the one with the
   highest target validation accuracy is selected; if none is admissible, the model with the
   lowest privacy-head accuracy is returned.

## Repository layout

```
src/
  run_cutclean.py              driver implementing the full procedure (entry point)
  pipeline.py                  training and evaluation loops, privacy-head routines, logging
  config.py                    argument parsing, defaults, deterministic seeding
  train.py                     backbone construction and privacy-head instrumentation
  dataloaders.py               dataset dispatch
  data_setup.py                dataset download, checksum verification and extraction
  pruning.py                   structured pruning and sparsity measurement
  save_and_load.py             checkpoint helpers
  datasets/, CorruptedCifarUnbiased.py, waterbirds_dataloader.py
  irene/                       privacy heads, mutual-information proxy, forward hooks
  model_architectures/         ResNet without skip connections, transformer MLP pruning
```

The driver is executed from the repository root as `python src/run_cutclean.py`, which places
`src/` on the import path.

## Requirements

Python 3.9 with PyTorch 2.5 and CUDA. Install the dependencies with

```bash
pip install -r requirements.txt
```

Weights & Biases logging is enabled by default and can be disabled by exporting
`WANDB_MODE=disabled`.

## Data

Three datasets are supported. In every case the target and the private attribute are
balanced, so that privacy leakage cannot be explained by a spurious correlation in the
training set; the construction is described in the supplementary material of the paper.

| Dataset | `--dataset` | Target | Private attribute |
| --- | --- | --- | --- |
| CelebA | `celeba-Blond_Hair-Male` | blond hair | gender |
| CelebA | `celeba-Heavy_Makeup-Male` | heavy make-up | gender |
| Corrupted-CIFAR10 | `corruptedcifarunbiased` | object class | corruption type |
| Waterbirds | `unbiasedWaterbirds` | waterbird / landbird | background |

Corrupted-CIFAR10 and Waterbirds are distributed as archives through
[imDalton/cutclean-datasets](https://huggingface.co/datasets/imDalton/cutclean-datasets)
and are retrieved with

```bash
python src/data_setup.py                 # into $CUTCLEAN_DATA, or ./data
python src/data_setup.py --check         # report what is already present
```

The archives are downloaded from the Hugging Face Hub, verified against their SHA-256
checksum and extracted once; a dataset that is already present is left untouched. Every
path resolves from `CUTCLEAN_DATA`, falling back to `./data` in the repository root, and can
still be overridden per run with `--datapath`, `--corruptedcifarunbiased_root` and
`--waterbirds_root`.

CelebA is **not** redistributed: its licence restricts research use and forbids
redistribution of the images. Obtain it from the official page
(https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) or any complete mirror, and place the
aligned images and annotations under `<CUTCLEAN_DATA>/celeba/`:

```
celeba/
  img_align_celeba/          202,599 aligned face images
  list_attr_celeba.txt       attribute annotations (.csv is also accepted)
  list_eval_partition.txt    official train/val/test partition (.csv is also accepted)
```

`datasets/celebA.py` reconstructs the balanced subset with a seeded per-group subsampling.
The split manifests published alongside the archives list the exact subsets (`seed`,
`split`, `image_id`, `target`, `bias`) so the selection can be audited independently. An
automatic download path exists in the loader, but it relies on a third-party mirror that is
currently unavailable, so the manual setup above is the supported route.

The resulting subset sizes reproduce those reported in the supplementary material: 5,548 /
728 / 720 for CelebA blond hair, 812 / 36 / 88 for heavy make-up, 8,900 / 2,500 / 1,300 for
Corrupted-CIFAR10 and 2,328 / 664 / 332 for Waterbirds.



## Running a single configuration

```bash
python src/run_cutclean.py \
    --dataset celeba-Blond_Hair-Male \
    --datapath /path/to/data/ \
    --projectName my_run \
    --gamma_list 0,0.01,0.1,1,5,10,50,100,500,1000 \
    --sparsity_list 0,0.05,0.1,0.15,0.2 \
    --t_ph 0.65 \
    --e_pre 400 --e_ft 10
```

`--e_task` controls an optional task-only pre-training stage that precedes the MI phase. It
defaults to the value of `--nb_epochs`; the experiment launchers set it to 0, so that
MI-aware training starts directly from the ImageNet weights, as in the paper.
`--reload_task_model` and `--reload_mi_model` restart an interrupted campaign from a later
stage. `python src/run_cutclean.py --help` lists the driver options followed by the general
ones.

## Outputs

Each run writes a timestamped directory under `models/<projectName>/`:

| File | Content |
| --- | --- |
| `pretrained_model.pth` | backbone before the MI phase |
| `mi_model.pth` | unpruned backbone trained with `γ*` |
| `pruned_model.pth` | selected pruned model |
| `manifest.json` | `γ*`, selected sparsity, selection rationale, paths above |
| `stats/gamma_selection.csv` | `γ` against the validation accuracy of the last privacy head |
| `stats/results_sparsity_sweep.csv` | per sparsity level: measured weight and channel sparsity, target and privacy-head accuracy on train, validation and test |
| `stats/run_summary.txt` | configuration and stage-by-stage log |


## Citation

```bibtex
@inproceedings{magliolo2026cutclean,
  author    = {Magliolo, Leonardo and Pastore, Vito Paolo and Valenzise, Giuseppe and Tartaglione, Enzo},
  title     = {CutClean: Neural Network Pruning for Privacy-Preserving Inference},
  booktitle = {Pattern Recognition -- 28th International Conference on Pattern Recognition, {ICPR} 2026, Lyon, France, August 17--22, 2026, Proceedings},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer Nature Switzerland},
  year      = {2026},
  pages     = {450--465},
  doi       = {10.1007/978-3-032-31452-9_30},
  isbn      = {978-3-032-31452-9}
}
```

## Acknowledgements and licence

The privacy heads and the mutual-information proxy in `src/irene/` are derived from the
reference implementation of *Information Removal at the bottleneck in Deep Neural Networks*
(E. Tartaglione, BMVC 2022, [arXiv:2210.00891](https://arxiv.org/abs/2210.00891)), released
under the MIT licence and reproduced here under the same terms; see `LICENSE`.

This work is supported by Hi! PARIS and by the ANR/France 2030 programme
(ANR-23-IACL-0005).
