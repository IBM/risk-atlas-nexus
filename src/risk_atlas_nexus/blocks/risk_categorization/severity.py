from typing import List, Dict
from risk_atlas_nexus.blocks.inference import InferenceEngine
from risk_atlas_nexus.blocks.prompt_templates import (
    RISK_SEVERITY_INSTRUCTION,
    RISK_SEVERITY_TEMPLATE,
)
from risk_atlas_nexus.blocks.inference.response_schema import RISK_CATEGORY_SCHEMA
from jinja2 import Template
from risk_atlas_nexus.toolkit.logging import configure_logger


logger = configure_logger(__name__)


class RiskSeverity:

    def __init__(self, inference_engine):
        self.inference_engine = inference_engine

    def categorize(self, usecases, domains, ai_tasks, ai_users, ai_subjects):
        """Categorize risk severity from a usecase description

        Args:
            usecases (List[str]):
                A List of strings describing AI usecases
            inference_engine (InferenceEngine):
                An LLM inference engine to identify AI tasks from usecases.

        Returns:
            List[Dict]:
                List of result containing risk categorisation information
        """
        # Prepare inference prompts
        messages = [
            [
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
            for usecase, domain, ai_task, ai_user, ai_subject in zip(
                usecases, domains, ai_tasks, ai_users, ai_subjects
            )
        ]

        # Invoke inference service
        return [
            result.prediction
            for result in self.inference_engine.chat(
                messages=messages,
                response_format=RISK_CATEGORY_SCHEMA,
                postprocessors=["json_object"],
                verbose=False,
            )
        ]
