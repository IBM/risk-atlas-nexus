"""
This file utilizes prompt templates that have been adapted from the source provided
here: https://github.com/sanja7s/AIDesign/blob/main/RiskGen_AI_Design.ipynb. The
RISK_SEVERITY_INSTRUCTION and RISK_SEVERITY_TEMPLATE prompt templates used below follow
the same instructions as given in the above link but have been modified to align with the
AI Act's risk-based categories shared here:
https://researchsummit.ie/wp-content/uploads/2022/10/AIRO-Delaram-Golpayegani.pdf
"""

from typing import List

from jinja2 import Template

from risk_atlas_nexus.blocks.prompt_response_schema import RISK_CATEGORY_SCHEMA
from risk_atlas_nexus.blocks.prompt_templates import (
    RISK_SEVERITY_INSTRUCTION,
    RISK_SEVERITY_TEMPLATE,
)
from risk_atlas_nexus.toolkit.logging import configure_logger


logger = configure_logger(__name__)


class RiskSeverityCategorizer:

    def __init__(self, inference_engine):
        self.inference_engine = inference_engine

    def categorize(
        self,
        domain: List[str],
        purpose: str,
        capability: str,
        ai_user: str,
        ai_subject: str,
    ):
        """Categorize the severity of risks associated with an AI system based on the specified parameters.

        Args:
            domain (str):
                Domain of the AI system
            purpose (str):
                Intended purpose of the AI system
            capability (str):
                The capability of an AI system to do what it is designed to do.
            ai_user (str):
                An AI user who interacts with the AI system
            ai_subject (str):
                AI subject impacted by the AI system

        Returns:
            Dict: Risk categorisation information
        """

        # Prepare a risk categorization inference message to be used with the inference engine.
        messages = [
            {
                "role": "system",
                "content": Template(RISK_SEVERITY_INSTRUCTION).render(),
            },
            {
                "role": "user",
                "content": Template(RISK_SEVERITY_TEMPLATE).render(
                    domain=domain,
                    purpose=purpose,
                    capability=capability,
                    aiUser=ai_user,
                    aiSubject=ai_subject,
                ),
            },
        ]

        # Invoke inference service
        return [
            result.prediction
            for result in self.inference_engine.chat(
                messages=[messages],
                response_format=RISK_CATEGORY_SCHEMA,
                postprocessors=["json_object"],
                verbose=False,
            )
        ][0]
