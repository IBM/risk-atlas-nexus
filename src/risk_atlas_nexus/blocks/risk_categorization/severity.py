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
        usecase: str,
        domain: List[str],
        ai_task: str,
        ai_user: str,
        ai_subject: str,
    ):
        """Categorize the severity of risks based on the use case description.

        Args:
            usecase (str):
                A usecase description
            domain (str):
                Domain type of usecase
            ai_task (List[str]):
                AI tasks inferred from usercase
            ai_user (str):
                AI user inferred from usercase
            ai_subject (str):
                AI subject inferred from  usercase

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
                    userIntent=usecase,
                    domain=domain,
                    aiTasks=ai_task,
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
