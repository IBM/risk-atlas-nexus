from abc import ABC, abstractmethod

from sssom_schema import Mapping

from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from risk_atlas_nexus.blocks.inference.base import InferenceEngine
from risk_atlas_nexus.metadata_base import MappingMethod
from risk_atlas_nexus.toolkit.logging import configure_logger


logger = configure_logger(__name__)


class RiskMappingBase(ABC):

    def __init__(
        self,
        new_risks: list[Risk],
        existing_risks: list[Risk],
        inference_engine: InferenceEngine,
        new_prefix: str,
        mapping_method: MappingMethod,
    ):
        self.inference_engine = inference_engine
        self._new_risks = new_risks
        self._existing_risks = existing_risks
        self._new_prefix = new_prefix
        self._mapping_method = mapping_method

    @abstractmethod
    def generate(
        self,
        new_risks: list[Risk],
        existing_risks: list[Risk],
        inference_engine: InferenceEngine,
        new_prefix: str,
        mapping_method: MappingMethod,
    ) -> list[Mapping]:
        raise NotImplementedError
