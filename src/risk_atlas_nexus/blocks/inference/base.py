from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from jinja2 import Template
from risk_atlas_nexus.blocks.inference.params import (
    InferenceEngineCredentials,
    RITSInferenceEngineParams,
    TextGenerationInferenceOutput,
    WMLInferenceEngineParams,
    OllamaInferenceEngineParams,
    VLLMInferenceEngineParams,
    OpenAIChatCompletionMessageParam,
)
from risk_atlas_nexus.toolkit.logging import configure_logger
import pydantic

logger = configure_logger(__name__)


class InferenceEngine(ABC):

    def __init__(
        self,
        model_name_or_path: str,
        credentials: Optional[Union[Dict, InferenceEngineCredentials]] = None,
        parameters: Optional[
            Union[
                RITSInferenceEngineParams,
                WMLInferenceEngineParams,
                OllamaInferenceEngineParams,
                VLLMInferenceEngineParams,
            ]
        ] = None,
        postprocessors: List[str] = None,
        concurrency_limit: int = 10,
    ):
        self.model_name_or_path = model_name_or_path
        self.parameters = self._check_if_parameters_are_valid(parameters)
        self.client = self.create_client(
            self.prepare_credentials(credentials if credentials else {})
        )
        self.postprocessors = postprocessors
        self.concurrency_limit = concurrency_limit

        logger.info(f"Created {self._inference_engine_type} inference engine.")

    def _check_if_parameters_are_valid(self, parameters):
        if parameters:
            invalid_params = []
            for param_key, _ in parameters.items():
                if param_key not in list(
                    self._inference_engine_parameter_class.__annotations__
                ):
                    invalid_params.append(param_key)

            if len(invalid_params) > 0:
                raise Exception(
                    f"Invalid parameters found: {invalid_params}. {self._inference_engine_type} inference engine only supports {list(self._inference_engine_parameter_class.__annotations__)}"
                )

        return parameters

    def prepare_prompt(self, prompt_template: str, usecase: str, **kwargs) -> List[str]:
        return Template(prompt_template).render(
            usecase=usecase,
            **kwargs,
        )

    def _to_openai_format(self, prompt: Union[OpenAIChatCompletionMessageParam, str]):
        if isinstance(prompt, str):
            return [{"role": "user", "content": prompt}]
        elif pydantic.TypeAdapter(OpenAIChatCompletionMessageParam).validate_python(
            prompt
        ):
            return prompt
        else:
            raise Exception(
                f"Invalid input format: {prompt}. Please use openai format or plain str."
            )

    @abstractmethod
    def prepare_credentials(
        self,
        credentials: Union[Dict, InferenceEngineCredentials],
    ) -> InferenceEngineCredentials:
        raise NotImplementedError

    @abstractmethod
    def create_client(self, credentials: InferenceEngineCredentials) -> Any:
        raise NotImplementedError

    @abstractmethod
    def generate(
        self, prompts: List[str], response_format=None, verbose=True
    ) -> List[TextGenerationInferenceOutput]:
        raise NotImplementedError

    @abstractmethod
    def chat(
        self,
        prompts: Union[
            List[OpenAIChatCompletionMessageParam],
            List[str],
        ],
        response_format=None,
        verbose=True,
    ) -> List[TextGenerationInferenceOutput]:
        raise NotImplementedError
