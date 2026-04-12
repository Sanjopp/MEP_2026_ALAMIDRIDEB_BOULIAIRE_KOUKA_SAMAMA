"""
Module de chargement de fichiers depuis S3 ou disque local.
"""

from io import BytesIO
import logging
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import torch

logger = logging.getLogger(__name__)


def _get_s3_client():
    """
    Crée le client S3 à la demande.
    Lit automatiquement AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY
    depuis les variables d'environnement.
    """
    return boto3.client("s3")


def load_bytes_from_s3(bucket: str, key: str) -> bytes:
    """
    Télécharge un fichier binaire brut depuis S3.
    Utilisé pour les formats IDX (FashionMNIST).
    """
    logger.info("Chargement bytes depuis s3://%s/%s", bucket, key)
    try:
        s3 = _get_s3_client()
        obj = s3.get_object(Bucket=bucket, Key=key)
        return obj["Body"].read()
    except (BotoCoreError, ClientError) as exc:
        raise RuntimeError(f"Erreur S3 lors du chargement de s3://{bucket}/{key}") from exc


def save_bytes_to_local(data: bytes, path: str | Path) -> None:
    """
    Sauvegarde des bytes dans un fichier local.
    Crée les répertoires s'ils n'existent pas.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    logger.info("Fichier sauvegardé localement : %s", path)


def load_bytes_from_local(path: str | Path) -> bytes:
    """
    Charge des bytes depuis le disque local (pour IDX files).
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    logger.info("Chargement bytes local depuis %s", path)
    with open(path, "rb") as f:
        return f.read()


def load_tensor_from_s3(bucket: str, key: str, local_path: str | Path | None = None) -> dict:
    """
    Charge un fichier .pt depuis S3 et le sauvegarde localement.
    Utilisé pour CIFAR-10.
    """
    logger.info("Chargement .pt depuis s3://%s/%s", bucket, key)
    try:
        data = load_bytes_from_s3(bucket, key)
    except RuntimeError:
        raise

    tensor = torch.load(BytesIO(data), map_location="cpu", weights_only=True)

    # Sauvegarder localement si local_path fourni
    if local_path:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(tensor, local_path)
        logger.info("Fichier sauvegardé localement : %s", local_path)

    return tensor


def load_tensor_from_local(path: str | Path) -> dict:
    """
    Charge un fichier .pt depuis le disque local.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {path}")
    logger.info("Chargement local depuis %s", path)
    return torch.load(path, map_location="cpu", weights_only=True)


def load_artifact(
    bucket: str,
    s3_key: str,
    local_path: str | Path | None = None,
    force_local: bool = False,
) -> dict:
    """
    Charge un artefact .pt avec fallback automatique :
      1. Si force_local=True ou si local_path existe → local
      2. Sinon → S3 (télécharge et sauvegarde en local)
    """
    if force_local or (local_path and Path(local_path).exists()):
        return load_tensor_from_local(local_path)
    return load_tensor_from_s3(bucket, s3_key, local_path)


def load_idx_artifact(
    bucket: str,
    s3_key: str,
    local_path: str | Path | None = None,
    force_local: bool = False,
) -> bytes:
    """Charge fichier IDX avec sauvegarde automatique en local."""
    if force_local or (local_path and Path(local_path).exists()):
        return load_bytes_from_local(local_path)

    data = load_bytes_from_s3(bucket, s3_key)
    if local_path:
        save_bytes_to_local(data, local_path)
    return data
