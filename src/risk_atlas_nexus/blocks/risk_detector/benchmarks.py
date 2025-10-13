import json
from typing import List

from risk_atlas_nexus.ai_risk_ontology.datamodel.ai_risk_ontology import Risk
from risk_atlas_nexus.blocks.inference import TextGenerationInferenceOutput
from risk_atlas_nexus.blocks.prompt_response_schema import LIST_OF_STR_SCHEMA
from risk_atlas_nexus.blocks.risk_detector import RiskDetector


# Benchmark-specific risk identification template
BENCHMARK_RISK_IDENTIFICATION_TEMPLATE = """You are an expert at AI risk classification for AI benchmarks and evaluation datasets. Study the risks JSON below containing list of risk category and its description.

RISKS:
{{ risks }}

Instructions:
1. Identify the potential RISKS associated with the given benchmark or evaluation dataset. Consider how the benchmark might be misused, what biases it might contain, or what harmful capabilities it might measure or enable.
2. Focus on risks specific to benchmarks such as: dataset bias, capability enablement, misuse potential, evaluation gaps, and downstream application risks.
3. Use RISK `description` to verify if the risk is associated with the benchmark use case.
4. If the benchmark doesn't fit into any of the above RISKS categories, classify it as Unknown.
5. Respond with an array list{% if max_risk is not none %} (top {{ max_risk }} high risks categories){% endif %} of attribute 'category' containing the risk labels.
{% if cot_examples is not none and cot_examples|length > 0 %}
EXAMPLES:{% for example in cot_examples %}
Benchmark: {{ example.Usecase }}
Risks: {{ example.Risks }}{% endfor %}
===== END OF EXAMPLES ======
{% endif %}
Benchmark: {{ usecase }}
Risks: """


class BenchmarkRiskDetector(RiskDetector):

    def detect(self, usecases: List[str]) -> List[List[Risk]]:
        prompts = [
            self.prompt_builder(prompt_template=BENCHMARK_RISK_IDENTIFICATION_TEMPLATE).build(
                cot_examples=self._examples,
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
