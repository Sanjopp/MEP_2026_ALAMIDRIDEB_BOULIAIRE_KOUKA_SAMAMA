# mep-2026-alamidrideb-bouliaire-kouka-samama

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

Mise en production du projet d'Advanced ML

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mep_2026_alamidrideb_bouliaire_kouka_samama and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mep_2026_alamidrideb_bouliaire_kouka_samama   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mep_2026_alamidrideb_bouliaire_kouka_samama a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

## Données

### Où sont les données ?

Les datasets sont stockés sur S3 :

| Dataset      | Chemin S3                                          | Taille |
|--------------|----------------------------------------------------|--------|
| CIFAR-10     | `data-mep-project/DATA/Cifar_10/cifar10_uint8.pt`  | ~150MB |
| FashionMNIST | `data-mep-project/DATA/FashionMNIST/raw/`          | ~52MB  |

### Comment les récupérer ?

Pour l'installation et le chargement des données, voir [`data/README.md`](data/README.md).

## Variables d'environnement

Voir `config/secrets.env.example` pour la liste complète.

| Variable                | Description             | Défaut             |
|-------------------------|-------------------------|--------------------|
| `AWS_ACCESS_KEY_ID`     | Clé d'accès AWS         | —                  |
| `AWS_SECRET_ACCESS_KEY` | Clé secrète AWS         | —                  |
| `AWS_DEFAULT_REGION`    | Région du bucket        | `eu-north-1`       |
| `S3_BUCKET`             | Nom du bucket           | `data-mep-project` |
| `BATCH_SIZE`            | Taille de batch         | `128`              |
| `NUM_WORKERS`           | Workers DataLoader      | `0`                |