"""ModelLoader — loads XGBoost artifact and training stats from S3 at startup."""

from __future__ import annotations

import logging

from app.services.s3_helper import load_pickle_from_s3, parse_s3_uri

logger = logging.getLogger(__name__)


class ModelLoader:
    def __init__(self) -> None:
        self._model = None
        self._metadata: dict = {}
        self._training_stats: dict = {}
        self.loaded: bool = False

    def load_from_s3(self, s3_uri: str) -> None:
        """Load a pickled model artifact from S3.

        Expected s3_uri formats:
          s3://bucket/path/to/model.pkl          → loads model directly
          s3://bucket/path/to/bundle.pkl         → loads dict with keys
                                                    'model', 'metadata', 'training_stats'
        """
        bucket, key = parse_s3_uri(s3_uri)
        artifact = load_pickle_from_s3(bucket, key)

        if isinstance(artifact, dict):
            self._model = artifact.get("model")
            self._metadata = artifact.get("metadata", {})
            self._training_stats = artifact.get("training_stats", {})
        else:
            # bare model object (e.g. XGBClassifier saved directly)
            self._model = artifact

        self.loaded = True
        logger.info("Model loaded from %s (type=%s)", s3_uri, type(self._model).__name__)

    def get_model(self):
        return self._model

    def get_metadata(self) -> dict:
        return self._metadata

    def get_training_stats(self) -> dict:
        return self._training_stats
