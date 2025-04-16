import json

from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from risk_atlas_nexus.blocks.inference import TextGenerationInferenceOutput
from risk_atlas_nexus.blocks.inference.templates import RISK_IDENTIFICATION_TEMPLATE
from risk_atlas_nexus.blocks.risk_detector import RiskDetector
from risk_atlas_nexus.blocks.inference.response_schema import LIST_OF_STR_SCHEMA


class GenericRiskDetector(RiskDetector):

    def detect(self, usecases: list[str]) -> list[Risk]:
        prompts = [
            self.inference_engine.prepare_prompt(
                prompt_template=RISK_IDENTIFICATION_TEMPLATE,
                usecase=usecase,
                risks=json.dumps(
                    [
                        {"category": risk.name, "description": risk.description}
                        for risk in self._risks
                    ],
                    indent=4,
                ),
                examples=self._examples,
            )
            for usecase in usecases
        ]

        LIST_OF_STR_SCHEMA["items"]["enum"] = [risk.name for risk in self._risks]
        inference_response = self.inference_engine.generate(
            prompts,
            response_format=LIST_OF_STR_SCHEMA,
            postprocessors=["list_of_str"],
        )

        return [
            list(
                filter(
                    lambda risk: risk.name in inference.prediction,
                    self._risks,
                )
            )
            for inference in inference_response
        ]
