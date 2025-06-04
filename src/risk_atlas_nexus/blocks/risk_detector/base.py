from abc import ABC, abstractmethod
from typing import List, Optional

from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import (
    Container,
    Risk,
    RiskTaxonomy,
)
from risk_atlas_nexus.blocks.inference.base import InferenceEngine
from risk_atlas_nexus.data import load_resource
from risk_atlas_nexus.toolkit.logging import configure_logger


logger = configure_logger(__name__)


RISK_IDENTIFICATION_EXAMPLES = load_resource("risk_generation_cot.json")


class RiskDetector(ABC):

    def __init__(
        self,
        ontology: Container,
        inference_engine: InferenceEngine,
        taxonomy: Optional[str] = None,
        max_risk: Optional[int] = None,
    ):
        self.inference_engine = inference_engine
        self._ontology = ontology
        self._taxonomy_id = taxonomy if taxonomy else "ibm-risk-atlas"
        self._risks = self.get_risks_by_taxonomy_id(ontology, self._taxonomy_id)
        self._examples = RISK_IDENTIFICATION_EXAMPLES.get(self._taxonomy_id, [])
        self.max_risk = max_risk

    def get_risks_by_taxonomy_id(
        self, ontology: Container, taxonomy_id: Optional[str] = None
    ):
        taxonomies = list(
            filter(
                lambda taxonomy: taxonomy.id == taxonomy_id,
                ontology.taxonomies,
            )
        )

        if len(taxonomies) > 0:
            taxonomy: RiskTaxonomy = taxonomies[0]
            logger.info(
                f"Selected taxonomy is {str(taxonomy.name)}. For more info: {taxonomy.url}"
            )

            return list(
                filter(
                    lambda risk: risk.isDefinedByTaxonomy == taxonomy.id,
                    ontology.risks,
                )
            )
        else:
            raise Exception(
                f"Risk Taxonomy: {taxonomy_id} not found. Available taxonomies: {[taxonomy.id for taxonomy in ontology.taxonomies]}"
            )

    @abstractmethod
    def detect(self, usecases: List[str]) -> List[List[Risk]]:
        raise NotImplementedError
