"""S3 helper utilities — upload/download files and pickle objects."""

from __future__ import annotations

import io
import logging
import pickle
from pathlib import Path

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)


def _client(region: str = "ap-northeast-1"):
    return boto3.client("s3", region_name=region)


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """'s3://bucket/key/path' → ('bucket', 'key/path')"""
    without_scheme = s3_uri.removeprefix("s3://")
    bucket, _, key = without_scheme.partition("/")
    return bucket, key


def upload_file(local_path: str | Path, bucket: str, key: str, region: str = "ap-northeast-1") -> None:
    """Upload a local file to S3."""
    try:
        _client(region).upload_file(str(local_path), bucket, key)
        logger.info("Uploaded %s → s3://%s/%s", local_path, bucket, key)
    except (BotoCoreError, ClientError) as e:
        logger.error("S3 upload failed: %s", e)
        raise


def download_file(bucket: str, key: str, local_path: str | Path, region: str = "ap-northeast-1") -> None:
    """Download a file from S3 to a local path."""
    try:
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        _client(region).download_file(bucket, key, str(local_path))
        logger.info("Downloaded s3://%s/%s → %s", bucket, key, local_path)
    except (BotoCoreError, ClientError) as e:
        logger.error("S3 download failed: %s", e)
        raise


def load_pickle_from_s3(bucket: str, key: str, region: str = "ap-northeast-1") -> object:
    """Load a pickle object directly from S3 into memory (no temp file)."""
    try:
        obj = _client(region).get_object(Bucket=bucket, Key=key)
        return pickle.loads(obj["Body"].read())  # noqa: S301
    except (BotoCoreError, ClientError) as e:
        logger.error("S3 pickle load failed: %s", e)
        raise


def save_pickle_to_s3(obj: object, bucket: str, key: str, region: str = "ap-northeast-1") -> None:
    """Pickle an object and upload it directly to S3 (no temp file)."""
    try:
        buf = io.BytesIO(pickle.dumps(obj))
        _client(region).upload_fileobj(buf, bucket, key)
        logger.info("Pickled object → s3://%s/%s", bucket, key)
    except (BotoCoreError, ClientError) as e:
        logger.error("S3 pickle save failed: %s", e)
        raise
