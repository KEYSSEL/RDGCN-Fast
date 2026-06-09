# RDGCN-Fast: Une Implémentation Optimisée de RDGCN pour l'Alignement d'Entités

Ce dépôt contient une implémentation optimisée et étendue du modèle **RDGCN (Relation-aware Dual-GAT)** pour l'alignement d'entités entre graphes de connaissances. Cette version inclut plusieurs améliorations pour accélérer l'entraînement et améliorer les performances, notamment l'initialisation sémantique et l'intégration d'attributs.

## Fonctionnalités

- **Modèle RDGCN-fast :** Implémentation du modèle qui utilise un GNN double (primal et dual) pour capturer la structure des entités et des relations.
- **Initialisation Sémantique :** Utilisation de **FastText** pour initialiser les embeddings, capturant ainsi la sémantique et la syntaxique des entités et d'attributs.
- **Intégration des Attributs :** Capacité à enrichir les représentations d'entités avec leurs attributs textuels pour des embeddings plus riches.
- **Échantillonnage Négatif Avancé :** Inclut des stratégies comme le *Nearest-Neighbor Sampling* pour générer des exemples négatifs plus pertinents et difficiles.
- **Performances Optimisées :** Utilisation de PyTorch et de calculs par lots (batching) pour accélérer significativement l'entraînement et l'évaluation.
- **Early Stopping :** Mécanisme de patience pour arrêter l'entraînement lorsque les performances stagnent, évitant le sur-apprentissage.

## Installation

### Prérequis

Le code a été testé avec les versions suivantes. Une configuration avec `conda` est recommandée pour gérer les dépendances.

- **Python :** `3.8`
- **CUDA :** `11.3`
- **PyTorch :** `1.10.0`
- **torch-geometric** et ses dépendances (scatter, sparse, etc.)
- **graph-tool**
- **gensim**
- **faiss-cpu**
- **scikit-learn**
- **numpy**
- **scipy**
- **igraph**
- **pandas**
- **tensorboard :** `2.14.0`

### Étapes d'installation avec Conda

1.  **Créez un nouvel environnement Conda :**
    ```bash
    conda create -n eakit_dev python=3.8
    conda activate eakit_dev
    ```

2.  **Installez `graph-tool` :**
    Cette bibliothèque peut être délicate à installer. La méthode la plus simple est via le canal `conda-forge`.
    ```bash
    conda install -y -c conda-forge graph-tool
    ```

3.  **Installez PyTorch avec le support CUDA :**
    Assurez-vous d'adapter la version de CUDA à votre configuration matérielle.
    ```bash
    pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
    ```

4.  **Installez les autres dépendances :**
    ```bash
    pip install pandas scikit-learn igraph torch-geometric torch-scatter torch-sparse torch-cluster torch-spline-conv gensim
    conda install -c conda-forge faiss-cpu tensorboard
    ```



## Préparation des Données

Le modèle attend une structure de dossier spécifique pour les jeux de données. Placez vos données dans un dossier (par exemple, `data/MY_DATASET`) avec les fichiers suivants :

```
/data/MY_DATASET/
├── ent_ids_1         # Mappage URI -> ID pour le KG1
├── ent_ids_2         # Mappage URI -> ID pour le KG2
├── rel_ids_1         # Mappage URI -> ID pour les relations du KG1
├── rel_ids_2         # Mappage URI -> ID pour les relations du KG2
├── triples_1         # Triplets (h, r, t) du KG1
├── triples_2         # Triplets (h, r, t) du KG2
├── all_pairs.txt     # Toutes les paires d'entités alignées (ID1, ID2)
└── (optionnel) attr_triples_1 # Triplets d'attributs (entité, attribut, valeur) pour le KG1
└── (optionnel) attr_triples_2 # Triplets d'attributs (entité, attribut, valeur)pour le KG2
```

## Comment Lancer l'Entraînement

Le script `run_RDGCN_fast.sh` fournit un exemple complet pour lancer une expérience. Vous pouvez le modifier ou exécuter `run.py` directement avec les arguments de votre choix.

```bash
./run_RDGCN_fast.sh
```

### Arguments Clés

- `--data_dir` : Chemin vers le dossier du jeu de données (ex: `./data/EN_FR_100K`).
- `--save` : Dossier pour sauvegarder les résultats (embeddings, logs, etc.).
- `--encoder "rdgcn"` : Spécifie l'utilisation du modèle RDGCN.
- `--hiddens "300,300,300"` : Dimensions des couches de l'encodeur.
- `--use_fasttext` : Active l'initialisation sémantique avec FastText.
- `--use_attr` : Active l'utilisation des attributs des entités.
- `--attr_alpha` : Poids accordé aux embeddings d'attributs lors de la fusion.
- `--sampling "N"` : Stratégie d'échantillonnage négatif (`N`: Nearest-Neighbor, `R`: Random, `T`: Typed).
- `--k "25"` : Nombre d'échantillons négatifs par triplet positif.
- `--early`  `--patience 3` : Active l'early stopping avec une patience de 3 `checks`.

## Fichiers de Sortie

Après une exécution, les fichiers suivants seront générés dans le dossier spécifié par `--save`:

- `*_enh_ins.npy` : Embeddings d'entités finaux (après l'encodeur GNN).
- `*_ins.npy` : Embeddings d'entités de base (appris).
- `*_rel.npy` : Embeddings de relations.
- `best_model.ckpt` : Poids du meilleur modèle GNN sauvegardé grâce à l'early stopping.
- `config.json` : Fichier de configuration JSON de l'exécution.
- `métriques.txt` : Résultats finaux (Hits@k, MR, MRR).
- `final_alignments_top10.txt` : Top 10 des prédictions d'alignement pour chaque entité du jeu de test.
- `uri_fasttext.model` : Modèle FastText entraîné (si `--use_fasttext` est activé).

