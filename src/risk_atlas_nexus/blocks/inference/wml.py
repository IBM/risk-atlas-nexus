import os
from typing import Any, Dict, List, Optional, Union

from dotenv import load_dotenv

from risk_atlas_nexus.blocks.inference.base import InferenceEngine
from risk_atlas_nexus.blocks.inference.params import (
    InferenceEngineCredentials,
    OpenAIChatCompletionMessageParam,
    TextGenerationInferenceOutput,
    WMLInferenceEngineParams,
)
from risk_atlas_nexus.blocks.inference.postprocessing import postprocess
from risk_atlas_nexus.metadata_base import InferenceEngineType
from risk_atlas_nexus.toolkit.job_utils import run_parallel
from risk_atlas_nexus.toolkit.logging import configure_logger


logger = configure_logger(__name__)

# load .env file to environment
load_dotenv()


class WMLInferenceEngine(InferenceEngine):

    _inference_engine_type = InferenceEngineType.WML
    _inference_engine_parameter_class = WMLInferenceEngineParams

    def prepare_credentials(
        self, credentials: Union[Dict, InferenceEngineCredentials]
    ) -> InferenceEngineCredentials:
        api_key = credentials.get(
            "api_key", os.environ.get(f"{self._inference_engine_type}_API_KEY", None)
        )
        assert api_key, (
            f"Error while trying to create {self._inference_engine_type}. "
            f"Please set the env variable: '{self._inference_engine_type}_API_KEY' or pass api_key to credentials."
        )

        api_url = credentials.get(
            "api_url", os.environ.get(f"{self._inference_engine_type}_API_URL", None)
        )
        assert api_url, (
            f"Error while trying to create {self._inference_engine_type}. "
            f"Please set the env variable: '{self._inference_engine_type}_API_URL' or pass api_url to credentials."
        )

        space_id = credentials.get(
            "space_id",
            os.environ.get(f"{self._inference_engine_type}_SPACE_ID", None),
        )

        project_id = credentials.get(
            "project_id",
            os.environ.get(f"{self._inference_engine_type}_PROJECT_ID", None),
        )

        if not (space_id or project_id):
            raise ValueError(
                "Error while trying to create {self._inference_engine_type} inference engine. "
                "Either 'space_id' or 'project_id' need to be specified when using WMLInferenceEngine. "
                f"Please set the env variable: '{self._inference_engine_type}_SPACE_ID'/'{self._inference_engine_type}_PROJECT_ID' "
                "or pass space_id/project_id to credentials.",
            )
        elif space_id and project_id:
            logger.warning(
                "Either 'space_id' or 'project_id' need to be "
                "specified, however, both were found. 'WMLInferenceEngine' "
                "will use space_id by default."
            )

        return InferenceEngineCredentials(
            api_key=api_key, api_url=api_url, space_id=space_id, project_id=project_id
        )

    def create_client(self, credentials: InferenceEngineCredentials):
        from ibm_watsonx_ai import APIClient
        from ibm_watsonx_ai.foundation_models import ModelInference

        client = APIClient(
            {
                "url": credentials["api_url"],
                "apikey": credentials["api_key"],
            }
        )
        if credentials["space_id"]:
            client.set.default_space(credentials["space_id"])
        else:
            client.set.default_project(credentials["project_id"])

        # self.parameters.update(
        #     {"response_format": self._create_schema_format(response_format)}
        # )
        return ModelInference(
            model_id=self.model_name_or_path, api_client=client, params=self.parameters
        )

    @postprocess
    def generate(
        self,
        prompts: List[str],
        response_format=None,
        postprocessors=None,
        verbose=True,
    ) -> List[TextGenerationInferenceOutput]:
        responses = []
        for response in self.client.generate(
            prompt=prompts,
            params=self.parameters,
            concurrency_limit=self.concurrency_limit,
        ):
            responses.append(self._prepare_generation_output(response))

        return responses

    def _prepare_generation_output(
        self, response
    ) -> List[TextGenerationInferenceOutput]:
        return TextGenerationInferenceOutput(
            prediction=response["results"][0]["generated_text"],
            input_tokens=response["results"][0]["input_token_count"],
            output_tokens=response["results"][0]["generated_token_count"],
            stop_reason=response["results"][0]["stop_reason"],
            model_name_or_path=self.model_name_or_path,
            inference_engine=str(self._inference_engine_type),
        )

    @postprocess
    def chat(
        self,
        messages: Union[
            List[OpenAIChatCompletionMessageParam],
            List[str],
        ],
        response_format=None,
        postprocessors=None,
        verbose=True,
    ) -> TextGenerationInferenceOutput:

        def chat_response(messages):
            response = self.client.chat(
                messages=self._to_openai_format(messages),
                params=self.parameters,
            )
            return self._prepare_chat_output(response)

        return run_parallel(
            chat_response,
            messages,
            f"Inferring with {self._inference_engine_type}",
            self.concurrency_limit,
            verbose=verbose,
        )

    def _prepare_chat_output(self, response) -> List[TextGenerationInferenceOutput]:
        print(response["choices"][0]["message"]["content"])
        return TextGenerationInferenceOutput(
            prediction=response["choices"][0]["message"]["content"],
            input_tokens=response["usage"]["prompt_tokens"],
            output_tokens=response["usage"]["completion_tokens"],
            stop_reason=response["choices"][0]["finish_reason"],
            model_name_or_path=self.model_name_or_path,
            inference_engine=str(self._inference_engine_type),
        )

    def _create_schema_format(self, response_format):
        if response_format:
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": "RITS_schema",
                    "schema": response_format,
                },
            }
        else:
            return None
