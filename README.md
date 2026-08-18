# KEGCL

PyTorch implementation of Knowledge-Enhanced Graph Contrastive Learning for protein complex identification.

## Requirements

- Python 3.9
- PyTorch 1.12.1
- CUDA 11.3 for GPU training

```bash
pip install -r requirements.txt
```

## Data

Place the following files in `data/`:

```text
data/
├── biogrid.txt
├── collins.txt
├── DIP.txt
├── krogan14k.txt
├── series_matrix.txt
├── go_slim_mapping.tab.txt
└── golden_standard.txt
```

The four PPI networks are BioGRID, Collins, DIP, and Krogan14k. `series_matrix.txt` contains time-series gene expression profiles, `go_slim_mapping.tab.txt` contains GO annotations, and `golden_standard.txt` contains reference complexes used only for evaluation.

## Run

Run preprocessing, training, clustering, and evaluation on all datasets:

```bash
python run_all.py --config configs/default.yaml --gold data/golden_standard.txt --device cuda:0
```

Use `--device cpu` for CPU execution.

The stages can also be run separately:

```bash
python preprocess.py --config configs/default.yaml --all
python train.py --config configs/default.yaml --dataset bio --device cuda:0
python cluster.py --config configs/default.yaml --dataset bio
python evaluate.py --predictions artifacts/runs/bio/complexes.txt --gold data/golden_standard.txt --output artifacts/runs/bio/evaluation.json
```

Dataset names are `bio`, `col`, `dip`, and `k14`.

## Outputs

```text
artifacts/
├── data/<dataset>.pt
└── runs/<dataset>/
    ├── checkpoint-final.pt
    ├── embeddings.pt
    ├── embeddings.tsv
    ├── complexes.txt
    ├── evaluation.json
    └── metrics.jsonl
```

All experiment settings are defined in `configs/default.yaml`. Generated artifacts are excluded from version control.

## Citation

```bibtex
@article{qu2025kegcl,
  title={KEGCL: Knowledge-Enhanced Graph Contrastive Learning for Protein Complex Identification},
  author={Qu, Yanchen and Wang, Shilong and Cui, Hai and Li, Meilin and Zhang, Yijia},
  journal={IEEE Journal of Biomedical and Health Informatics},
  year={2025},
  publisher={IEEE}
}
```
