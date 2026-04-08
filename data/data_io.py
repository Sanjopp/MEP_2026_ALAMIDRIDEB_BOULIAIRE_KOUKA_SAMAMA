"""
Module de chargement de fichiers depuis S3 ou disque local.
"""
import logging
from io import BytesIO
from pathlib import Path
from typing import Union

import boto3
import torch
from botocore.exceptions import BotoCoreError, ClientError

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
        raise RuntimeError(
            f"Erreur S3 lors du chargement de s3://{bucket}/{key}"
        ) from exc


def load_tensor_from_s3(bucket: str, key: str) -> dict:
    """
    Charge un fichier .pt depuis S3.
    Utilisé pour CIFAR-10.
    """
    logger.info("Chargement .pt depuis s3://%s/%s", bucket, key)
    try:
        data = load_bytes_from_s3(bucket, key)
    except RuntimeError:
        raise
    return torch.load(BytesIO(data), map_location="cpu", weights_only=True)


def load_tensor_from_local(path: Union[str, Path]) -> dict:
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
    local_path: Union[str, Path, None] = None,
    force_local: bool = False,
) -> dict:
    """
    Charge un artefact .pt avec fallback automatique :
      1. Si force_local=True ou si local_path existe → local
      2. Sinon → S3
    """
    if force_local or (local_path and Path(local_path).exists()):
        return load_tensor_from_local(local_path)
    return load_tensor_from_s3(bucket, s3_key)