"""FastAPI application entry point with lifespan startup/shutdown."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from app.config import get_settings
from app.routers import audit, drift, explain, model, predict
from app.routers import (
    feature_store as feature_store_router,
    stream as stream_router,
    graph as graph_router,
    clusters as clusters_router,
    sequence as sequence_router,
    thresholds as thresholds_router,
    cases as cases_router,
    alerts as alerts_router,
    copilot as copilot_router,
    monitoring as monitoring_router,
)
from app.services.audit_logger import AuditLogger
from app.services.drift_detector import DriftDetector
from app.services.model_loader import ModelLoader
from app.services.predictor import XGBPredictor
from app.services.shap_explainer import SHAPExplainer
from app.services.feature_store import FeatureStore
from app.services.ensemble_scorer import EnsembleScorer
from app.services.stream_consumer import StreamConsumer
from app.services.graph_engine import GraphEngine
from app.services.identity_clusterer import IdentityClusterer
from app.services.sequence_scorer import SequenceScorer
from app.services.threshold_controller import ThresholdController
from app.services.case_manager import CaseManager
from app.services.alert_router import AlertRouter
from app.services.ai_copilot import AICopilot
from app.services.monitoring_system import MonitoringSystem

logger = logging.getLogger(__name__)


class AppState:
    # Existing
    model_loader: ModelLoader
    predictor: XGBPredictor
    shap_explainer: SHAPExplainer
    drift_detector: DriftDetector
    audit_logger: AuditLogger
    # New
    feature_store: FeatureStore
    ensemble_scorer: EnsembleScorer
    stream_consumer: StreamConsumer
    graph_engine: GraphEngine
    identity_clusterer: IdentityClusterer
    sequence_scorer: SequenceScorer
    threshold_controller: ThresholdController
    case_manager: CaseManager
    alert_router: AlertRouter
    ai_copilot: AICopilot
    monitoring_system: MonitoringSystem


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialise all services on startup; clean up on shutdown."""
    settings = get_settings()
    logger.info("Starting AML Fraud Detection API…")

    # ── Existing services ──────────────────────────────────────────────────
    state.model_loader = ModelLoader()
    try:
        state.model_loader.load_from_s3(settings.model_s3_uri)
        logger.info("Model loaded from %s", settings.model_s3_uri)
    except NotImplementedError:
        logger.warning("ModelLoader not yet implemented — running without model")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load model from S3: %s", exc)

    state.predictor = XGBPredictor(state.model_loader)
    state.shap_explainer = SHAPExplainer(state.model_loader)
    state.drift_detector = DriftDetector(state.model_loader)
    state.audit_logger = AuditLogger(settings.database_url)

    # ── Feature Store ──────────────────────────────────────────────────────
    state.feature_store = FeatureStore(
        redis_url=settings.redis_url,
        database_url=settings.database_url,
    )
    await state.feature_store.connect()

    # ── Ensemble Scorer ────────────────────────────────────────────────────
    state.ensemble_scorer = EnsembleScorer()

    # ── Alert Router ───────────────────────────────────────────────────────
    state.alert_router = AlertRouter(
        rate_limit_per_hour=settings.alert_rate_limit_per_hour,
        cooldown_seconds=settings.alert_cooldown_seconds,
    )

    # ── Stream Consumer ────────────────────────────────────────────────────
    state.stream_consumer = StreamConsumer(
        broker_type=settings.stream_broker_type,
        feature_store=state.feature_store,
        predictor=state.predictor,
        ensemble_scorer=state.ensemble_scorer,
        audit_logger=state.audit_logger,
        alert_router=state.alert_router,
    )
    state.stream_consumer.start()

    # ── Graph Engine ───────────────────────────────────────────────────────
    state.graph_engine = GraphEngine()

    # ── Identity Clusterer ─────────────────────────────────────────────────
    state.identity_clusterer = IdentityClusterer(
        high_threshold=settings.threshold_high_floor,
    )

    # ── Sequence Scorer ────────────────────────────────────────────────────
    state.sequence_scorer = SequenceScorer()

    # ── Threshold Controller ───────────────────────────────────────────────
    state.threshold_controller = ThresholdController()

    # ── Case Manager ───────────────────────────────────────────────────────
    state.case_manager = CaseManager(high_threshold=settings.threshold_high_floor)
    state.case_manager.set_alert_router(state.alert_router)

    # ── AI Copilot ─────────────────────────────────────────────────────────
    state.ai_copilot = AICopilot(case_manager=state.case_manager)

    # ── Monitoring System ──────────────────────────────────────────────────
    state.monitoring_system = MonitoringSystem(alert_router=state.alert_router)

    # Attach state to app for router access via request.app.state
    app.state.feature_store = state.feature_store
    app.state.stream_consumer = state.stream_consumer
    app.state.graph_engine = state.graph_engine
    app.state.identity_clusterer = state.identity_clusterer
    app.state.sequence_scorer = state.sequence_scorer
    app.state.threshold_controller = state.threshold_controller
    app.state.case_manager = state.case_manager
    app.state.alert_router = state.alert_router
    app.state.ai_copilot = state.ai_copilot
    app.state.monitoring_system = state.monitoring_system

    logger.info("All services initialised — API ready")
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────
    state.stream_consumer.stop()
    logger.info("Shutting down AML Fraud Detection API…")


def create_app() -> FastAPI:
    app = FastAPI(
        title="AML Fraud Detection API",
        description=(
            "Real-time fraud risk scoring, graph-based detection, identity clustering, "
            "behavioral profiling, adaptive thresholds, case management, AI copilot, "
            "cross-platform alerting, feature store, and unified monitoring."
        ),
        version="0.2.0",
        lifespan=lifespan,
    )

    # Existing routers
    app.include_router(predict.router, tags=["Prediction"])
    app.include_router(explain.router, tags=["Explainability"])
    app.include_router(drift.router, tags=["Drift"])
    app.include_router(audit.router, tags=["Audit"])
    app.include_router(model.router, tags=["Model"])

    # New routers
    app.include_router(feature_store_router.router)
    app.include_router(stream_router.router)
    app.include_router(graph_router.router)
    app.include_router(clusters_router.router)
    app.include_router(sequence_router.router)
    app.include_router(thresholds_router.router)
    app.include_router(cases_router.router)
    app.include_router(alerts_router.router)
    app.include_router(copilot_router.router)
    app.include_router(monitoring_router.router)

    @app.get("/health", include_in_schema=False)
    async def health() -> dict:
        """App Runner / ALB 健康檢查端點。"""
        return {"status": "ok", "version": app.version}

    @app.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
