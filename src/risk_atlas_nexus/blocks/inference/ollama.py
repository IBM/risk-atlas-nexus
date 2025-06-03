import os
from typing import Any, Dict, List, Union

from dotenv import load_dotenv

from risk_atlas_nexus.blocks.inference.base import InferenceEngine
from risk_atlas_nexus.blocks.inference.params import (
    InferenceEngineCredentials,
    OllamaInferenceEngineParams,
    OpenAIChatCompletionMessageParam,
    TextGenerationInferenceOutput,
)
from risk_atlas_nexus.blocks.inference.postprocessing import postprocess
from risk_atlas_nexus.metadata_base import InferenceEngineType
from risk_atlas_nexus.toolkit.job_utils import run_parallel
from risk_atlas_nexus.toolkit.logging import configure_logger


logger = configure_logger(__name__)

# load .env file to environment
load_dotenv()


class OllamaInferenceEngine(InferenceEngine):

    _inference_engine_type = InferenceEngineType.OLLAMA
    _inference_engine_parameter_class = OllamaInferenceEngineParams

    def prepare_credentials(
        self, credentials: Union[Dict, InferenceEngineCredentials]
    ) -> InferenceEngineCredentials:
        api_url = credentials.get(
            "api_url",
            os.environ.get(f"{self._inference_engine_type}_API_URL", None),
        )
        assert api_url, (
            f"Error while trying to run {self._inference_engine_type}. "
            f"Please pass api_url to credentials or set the env variable: '{self._inference_engine_type}_API_URL'"
        )

        if api_url:
            logger.info(
                f"{self._inference_engine_type} inference engine will execute requests on the server at {api_url}."
            )

        return InferenceEngineCredentials(api_url=api_url)

    def create_client(self, credentials):
        from ollama import Client

        return Client(host=credentials["api_url"])

    @postprocess
    def generate(
        self,
        prompts: List[str],
        response_format=None,
        postprocessors=None,
        verbose=True,
    ) -> List[TextGenerationInferenceOutput]:
        def generate_text(prompt: str):
            response = self.client.generate(
                model=self.model_name_or_path,
                prompt=prompt,
                format=response_format,
                options=self.parameters,  # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
            )
            return self._prepare_prediction_output(response.response)

        return run_parallel(
            generate_text,
            prompts,
            f"Inferring with {self._inference_engine_type}",
            self.concurrency_limit,
            verbose=verbose,
        )

    @postprocess
    def chat(
        self,
        messages: Union[
            List[OpenAIChatCompletionMessageParam],
            List[str],
        ],
        tools=None,
        response_format=None,
        postprocessors=None,
        verbose=True,
    ) -> List[TextGenerationInferenceOutput]:

        def chat_response(messages):
            response = self.client.chat(
                model=self.model_name_or_path,
                messages=self._to_openai_format(messages),
                tools=tools,
                format=response_format,
                options=self.parameters,  # https://github.com/ollama/ollama/blob/main/docs/modelfile.md#valid-parameters-and-values
            )
            return self._prepare_prediction_output(response.message.content)

        return run_parallel(
            chat_response,
            messages,
            f"Inferring with {self._inference_engine_type}",
            self.concurrency_limit,
            verbose=verbose,
        )

    def _prepare_prediction_output(self, prediction):
        return TextGenerationInferenceOutput(
            prediction=prediction,
            model_name_or_path=self.model_name_or_path,
            inference_engine=str(self._inference_engine_type),
        )
