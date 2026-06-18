# RDGCN-Fast: An Optimized Implementation of RDGCN for Entity Alignment

This repository contains an optimized and extended implementation of the **RDGCN (Relation-aware Dual-GAT)** model for entity alignment between knowledge graphs. This version includes several improvements to speed up training and enhance performance, notably semantic initialization and attribute integration.

## Features

- **RDGCN-fast Model:** Implementation of the model using a dual GNN (primal and dual) to capture the structure of entities and relations.
- **Semantic Initialization:** Use of **FastText** to initialize embeddings, thereby capturing the semantics and syntax of entities and attributes.
- **Attribute Integration:** Ability to enrich entity representations with their textual attributes for richer embeddings.
- **Advanced Negative Sampling:** Includes strategies like *Nearest-Neighbor Sampling* to generate more relevant and hard negative examples.
- **Optimized Performance:** Use of PyTorch and batch calculations (batching) to significantly speed up training and evaluation.
- **Early Stopping:** Patience mechanism to stop training when performance stagnates, preventing overfitting.

## Installation

### Prerequisites

The code has been tested with the following versions. A configuration with `conda` is recommended to manage dependencies.

- **Python :** `3.8`
- **CUDA :** `11.3`
- **PyTorch :** `1.10.0`
- **torch-geometric** and its dependencies (scatter, sparse, etc.)
- **graph-tool**
- **gensim**
- **faiss-cpu**
- **scikit-learn**
- **numpy**
- **scipy**
- **igraph**
- **pandas**
- **tensorboard :** `2.14.0`

### Installation Steps with Conda

1.  **Create a new Conda environment:**
    ```bash
    conda create -n eakit_dev python=3.8
    conda activate eakit_dev
    ```

2.  **Install `graph-tool`:**
    This library can be tricky to install. The easiest method is via the `conda-forge` channel.
    ```bash
    conda install -y -c conda-forge graph-tool
    ```

3.  **Install PyTorch with CUDA support:**
    Make sure to adapt the CUDA version to your hardware configuration.
    ```bash
    pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```

4.  **Install other dependencies:**
    ```bash
    pip install pandas scikit-learn igraph torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv gensim
    conda install -c conda-forge faiss-cpu tensorboard
    ```



## Data Preparation

The model expects a specific folder structure for datasets. Place your data in a folder (e.g., `data/MY_DATASET`) with the following files:

```text
/data/MY_DATASET/
├── ent_ids_1         # URI-to-ID mapping for KG1 entities
├── ent_ids_2         # URI-to-ID mapping for KG2 entities
├── rel_ids_1         # URI-to-ID mapping for KG1 relations
├── rel_ids_2         # URI-to-ID mapping for KG2 relations
├── triples_1         # KG1 triples (h, r, t)
├── triples_2         # KG2 triples (h, r, t)
├── all_pairs.txt     # All aligned entity pairs (ID1, ID2)
└── (optional) attr_triples_1 # Attribute triples (entity, attribute, value) for KG1
└── (optional) attr_triples_2 # Attribute triples (entity, attribute, value) for KG2
```

## Running Training

The `run_RDGCN_fast.sh` script provides a complete example for launching an experiment. You can modify it or run `run.py` directly with your preferred arguments.

```bash
./run_RDGCN_fast.sh
```

### Key Arguments

* `--data_dir`: Path to the dataset directory (e.g., `./data/EN_FR_100K`).
* `--save`: Directory where outputs will be saved (embeddings, models, etc.).
* `--encoder "rdgcn"`: Specifies the use of the RDGCN-fast model.
* `--hiddens "300,300,300"`: Dimensions of the encoder layers.
* `--use_fasttext`: Enables semantic initialization using FastText.
* `--use_attr`: Enables the use of entity attributes.
* `--attr_alpha`: Weight assigned to attribute embeddings during fusion.
* `--sampling "N"`: Negative sampling strategy (`N`: Nearest Neighbor, `R`: Random, `T`: Typed).
* `--k "25"`: Number of negative samples per positive triple.
* `--early --patience 3`: Enables early stopping with a patience of 3 validation checks.

## Output Files

After training, the following files will be generated in the directory specified by `--save`:

* `*_enh_ins.npy`: Final entity embeddings (after the GNN encoder).
* `*_ins.npy`: Base entity embeddings (learned embeddings).
* `*_rel.npy`: Relation embeddings.
* `best_model.ckpt`: Weights of the best GNN model saved via early stopping.
* `config.json`: JSON configuration file for the experiment.
* `metrics.txt`: Final evaluation results (Hits@k, MR, MRR).
* `final_alignments_top10.txt`: Top-10 alignment predictions for each entity in the test set.
* `uri_fasttext.model`: Trained FastText model (if `--use_fasttext` is enabled).

```
```
