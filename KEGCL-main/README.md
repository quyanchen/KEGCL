# CLSTRA

This is the code for our paper: CLSTRA: Knowledge-Enhanced Spatiotemporal Graph Contrastive Learning with Randomized Architecture for Protein Complex Identification.

## **Requirements**

- torch  1.7.1
- torch-geometric 1.7.2
- sklearn 1.0.2
- numpy 1.21.6
- pyyaml 6.0.1
- nni 2.10.1

Install all dependencies using

```python
pip install -r requirements.txt
```

If you encounter some problems during installing `torch-geometric`, please refer to the installation manual on its [official website](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html).

## **Datasets**

| Dataset | Proteins | Edges |  Cliques | Average Neighbors |
| --- | --- | --- | --- | --- |
| Biogrid | 5640 | 59748 | 38616 | 21.187 |
| DIP | 4928 | 17201 | 7832 | 6.981 |
| Krogan14K | 3581 | 14076 | 4075 | 7.861 |

## **Usage**

### Construct Knowledge-Enhanced_PPI Network

Python Construction_of_Knowledge-Enhanced_PPI.py to Construct Knowledge-Enhanced_PPI Network

### Node Representation Learning

Python Node_representation_learning.py to achieve GCL of randomized architectures

### Core-Affiliate Clustering Algorithm

Python Core-Affiliate_Clustering.py to identify protein complexes

Use performance.py to evaluate performance.