from typing import Optional

from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Container
from risk_atlas_nexus.blocks.inference import InferenceEngine
from risk_atlas_nexus.toolkit.logging import configure_logger

from .base import RiskDetector
from .generic import GenericRiskDetector


logger = configure_logger(__name__)


class AutoRiskDetector:

    @classmethod
    def create(
        cls,
        ontology: Container,
        inference_engine: InferenceEngine,
        taxonomy: Optional[str] = None,
        max_risk: Optional[int] = None,
    ) -> RiskDetector:
        risk_detector = GenericRiskDetector(
            ontology, inference_engine, taxonomy, max_risk
        )
        return risk_detector
