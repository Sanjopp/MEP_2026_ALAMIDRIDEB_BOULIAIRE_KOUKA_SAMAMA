from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class S3Config:
    bucket: str
    region: str
    prefix: str  # e.g. "datasets"


@dataclass(frozen=True)
class DatasetConfig:
    s3_subdir: str
    local_subdir: str


def _repo_root() -> Path:
    # repo_root/src/data_io.py -> repo_root
    return Path(__file__).resolve().parents[1]


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        import yaml  # PyYAML (added later in requirements/environment)
    except ImportError as e:
        raise ImportError(
            "Missing dependency: PyYAML. It will be added to the project dependencies in a later commit."
        ) from e

    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid YAML structure in {path}")
    return data


def load_config(config_path: Optional[Path] = None) -> Tuple[S3Config, Dict[str, DatasetConfig]]:
    root = _repo_root()
    cfg_path = config_path or (root / "config" / "config.yaml")
    cfg = _load_yaml(cfg_path)

    s3 = cfg.get("s3", {})
    bucket = s3.get("bucket")
    region = s3.get("region")
    prefix = s3.get("prefix")

    if not bucket or not region or not prefix:
        raise ValueError("config/config.yaml must define s3.bucket, s3.region, s3.prefix")

    s3_cfg = S3Config(
        bucket=str(bucket),
        region=str(region),
        prefix=str(prefix).strip("/"),
    )

    datasets_raw = cfg.get("datasets")
    if not isinstance(datasets_raw, dict) or not datasets_raw:
        raise ValueError("config/config.yaml must define a non-empty datasets mapping")

    datasets: Dict[str, DatasetConfig] = {}
    for name, d in datasets_raw.items():
        if not isinstance(d, dict):
            raise ValueError(f"Dataset '{name}' must be a mapping")

        s3_subdir = d.get("s3_subdir")
        local_subdir = d.get("local_subdir")
        if not s3_subdir or not local_subdir:
            raise ValueError(f"Dataset '{name}' must define s3_subdir and local_subdir")

        datasets[str(name)] = DatasetConfig(
            s3_subdir=str(s3_subdir).strip("/"),
            local_subdir=str(local_subdir).strip("/"),
        )

    return s3_cfg, datasets


def _get_s3_client(region: str):
    try:
        import boto3  # added later in dependencies
    except ImportError as e:
        raise ImportError(
            "Missing dependency: boto3. It will be added to the project dependencies in a later commit."
        ) from e

    # Credentials come from environment variables (Codespaces/GitHub Secrets)
    return boto3.client("s3", region_name=region)


def download_prefix(bucket: str, prefix: str, local_dir: Path, region: str) -> int:
    """
    Download all objects under s3://bucket/prefix into local_dir, preserving relative paths.
    Returns the number of downloaded files.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    s3 = _get_s3_client(region)

    token: Optional[str] = None
    downloaded = 0

    while True:
        kwargs: Dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token

        resp = s3.list_objects_v2(**kwargs)
        contents = resp.get("Contents", [])

        for obj in contents:
            key = obj["Key"]
            if key.endswith("/"):
                continue

            rel = key[len(prefix) :].lstrip("/")
            dest = local_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Skip when file already matches size
            size = obj.get("Size")
            if dest.exists() and size is not None and dest.stat().st_size == size:
                continue

            s3.download_file(bucket, key, str(dest))
            downloaded += 1

        if resp.get("IsTruncated"):
            token = resp.get("NextContinuationToken")
        else:
            break

    return downloaded


def ensure_dataset_local(dataset_name: str, config_path: Optional[Path] = None) -> Path:
    """
    Ensure dataset is present locally under ./data/<local_subdir>.
    Downloads recursively from S3 when local directory is missing or empty.
    """
    s3_cfg, datasets = load_config(config_path=config_path)

    if dataset_name not in datasets:
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {sorted(datasets.keys())}")

    ds = datasets[dataset_name]

    root = _repo_root()
    data_root = root / "data"
    local_dir = data_root / ds.local_subdir
    local_dir.mkdir(parents=True, exist_ok=True)

    has_files = any(p.is_file() for p in local_dir.rglob("*"))
    if has_files:
        return local_dir

    full_prefix = f"{s3_cfg.prefix}/{ds.s3_subdir}".strip("/") + "/"
    download_prefix(
        bucket=s3_cfg.bucket,
        prefix=full_prefix,
        local_dir=local_dir,
        region=s3_cfg.region,
    )
    return local_dir
