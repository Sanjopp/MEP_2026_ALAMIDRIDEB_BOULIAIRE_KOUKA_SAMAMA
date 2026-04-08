# Data

## Datasets utilisés

| Dataset      | Taille images | Canaux | Classes | Train  | Test   |
|--------------|---------------|--------|---------|--------|--------|
| CIFAR-10     | 32×32         | 3 (RGB)| 10      | 50 000 | 10 000 |
| FashionMNIST | 28×28         | 1 (gris)| 10     | 60 000 | 10 000 |

---

## Stockage S3

Tous les datasets sont stockés sur S3 dans le bucket `data-mep-project` :

```
data-mep-project/
├── DATA/
│   ├── Cifar_10/
│   │   └── cifar10_uint8.pt              ← images + labels + mean + std
│   └── FashionMNIST/
│       └── raw/
│           ├── train-images-idx3-ubyte   ← 44.9 MB
│           ├── train-labels-idx1-ubyte   ← 58.6 KB
│           ├── t10k-images-idx3-ubyte    ←  7.5 MB
│           └── t10k-labels-idx1-ubyte    ←  9.8 KB
```

---

## Format des fichiers

### CIFAR-10 — `.pt`
Fichier PyTorch unique contenant un dictionnaire :
```python
{
    "X_train_uint8": Tensor(50000, 3072),  # uint8
    "y_train":       Tensor(50000,),       # long
    "X_test_uint8":  Tensor(10000, 3072),  # uint8
    "y_test":        Tensor(10000,),       # long
    "mean":          Tensor(3,),           # float32 par canal RGB
    "std":           Tensor(3,),           # float32 par canal RGB
}
```

### FashionMNIST — IDX binaire
Format binaire original de Yann LeCun :
```
Fichier images : [magic:4B][n:4B][rows:4B][cols:4B][pixels: n×rows×cols bytes]
Fichier labels : [magic:4B][n:4B][labels: n bytes]
```
Mean et std sont **calculés dynamiquement** depuis le train set au chargement.

---

## Structure du module

```
data/
├── __init__.py
├── data_io.py            ← chargement générique S3 / local
├── dataload_cifar.py     ← Dataset + DataLoader CIFAR-10
├── dataload_fashion.py   ← Dataset + DataLoader FashionMNIST
└── local/                ← cache local (non versionné, dans .gitignore)
    ├── cifar10/
    └── fashion/
```

---

## Charger les données

### Depuis S3 (mode normal)

```python
from dotenv import load_dotenv
load_dotenv()

from data.dataload_cifar import load_cifar10
from data.dataload_fashion import load_fashion_mnist

train_loader, test_loader = load_cifar10()
train_loader, test_loader = load_fashion_mnist()
```

### Mettre en cache local (recommandé pendant le dev)

Lance **une seule fois** pour éviter de retélécharger à chaque run :

```bash
# CIFAR-10
python -c "
from dotenv import load_dotenv; load_dotenv()
import boto3
s3 = boto3.client('s3')
s3.download_file(
    'data-mep-project',
    'DATA/Cifar_10/cifar10_uint8.pt',
    'data/local/cifar10/cifar10_uint8.pt'
)
print('CIFAR-10 ✅')
"

# FashionMNIST
python -c "
from dotenv import load_dotenv; load_dotenv()
import boto3
s3 = boto3.client('s3')
files = [
    'train-images-idx3-ubyte',
    'train-labels-idx1-ubyte',
    't10k-images-idx3-ubyte',
    't10k-labels-idx1-ubyte',
]
for f in files:
    s3.download_file(
        'data-mep-project',
        f'DATA/FashionMNIST/raw/{f}',
        f'data/local/fashion/{f}'
    )
    print(f'{f} ✅')
"
```

### Depuis le cache local

```python
train_loader, test_loader = load_cifar10(force_local=True)
train_loader, test_loader = load_fashion_mnist(force_local=True)
```

---

## Preprocessing

| Étape          | CIFAR-10                        | FashionMNIST                     |
|----------------|---------------------------------|----------------------------------|
| Stockage       | uint8 (0-255)                   | uint8 (0-255)                    |
| Normalisation  | à la volée dans `__getitem__`   | à la volée dans `__getitem__`    |
| Mean           | stockée dans le `.pt`           | calculée depuis le train set     |
| Std            | stockée dans le `.pt`           | calculée depuis le train set     |
| Shape sortie   | `(batch, 3, 32, 32)` float32    | `(batch, 1, 28, 28)` float32     |