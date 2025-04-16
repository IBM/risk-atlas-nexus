import json
import os
from abc import ABC, abstractmethod
from typing import Optional
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import (
    Container,
    RiskTaxonomy,
)
from risk_atlas_nexus.blocks.inference.base import InferenceEngine
from risk_atlas_nexus.data import get_templates_path
from risk_atlas_nexus.toolkit.logging import configure_logger


logger = configure_logger(__name__)


def load_risk_identification_examples():
    with open(
        os.path.join(get_templates_path(), "risks_identification_examples.json")
    ) as f:
        return json.load(f)


RISK_IDENTIFICATION_EXAMPLES = load_risk_identification_examples()


class RiskDetector(ABC):

    def __init__(
        self,
        ontology: Container,
        inference_engine: InferenceEngine,
        taxonomy: Optional[str] = None,
    ):
        self.inference_engine = inference_engine
        self._ontology = ontology
        self._taxonomy_id = taxonomy if taxonomy else "ibm-risk-atlas"
        self._risks = self.get_risks_by_taxonomy_id(ontology, self._taxonomy_id)
        self._examples = RISK_IDENTIFICATION_EXAMPLES.get(self._taxonomy_id, [])

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
    def detect(self, usecases: list[str]) -> list[Risk]:
        raise NotImplementedError
