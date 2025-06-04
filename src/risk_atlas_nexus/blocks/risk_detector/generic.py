import json
from typing import List

from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from risk_atlas_nexus.blocks.inference import TextGenerationInferenceOutput
from risk_atlas_nexus.blocks.prompt_builder import FewShotPromptBuilder
from risk_atlas_nexus.blocks.prompt_response_schema import LIST_OF_STR_SCHEMA
from risk_atlas_nexus.blocks.prompt_templates import RISK_IDENTIFICATION_TEMPLATE
from risk_atlas_nexus.blocks.risk_detector import RiskDetector


class GenericRiskDetector(RiskDetector):

    def detect(self, usecases: List[str]) -> List[List[Risk]]:
        prompts = [
            FewShotPromptBuilder(prompt_template=RISK_IDENTIFICATION_TEMPLATE).build(
                cot_examples=self._examples["cot_examples"],
                usecase=usecase,
                risks=json.dumps(
                    [
                        {"category": risk.name, "description": risk.description}
                        for risk in self._risks
                    ],
                    indent=4,
                ),
                max_risk=self.max_risk,
            )
            for usecase in usecases
        ]

        # Populate schema items
        json_schema = dict(LIST_OF_STR_SCHEMA)
        json_schema["items"]["enum"] = [risk.name for risk in self._risks]

        # Invoke inference service
        inference_response: List[TextGenerationInferenceOutput] = (
            self.inference_engine.generate(
                prompts,
                response_format=json_schema,
                postprocessors=["list_of_str"],
            )
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
