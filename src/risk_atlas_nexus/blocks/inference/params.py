import dataclasses
from typing import Any, Dict, List, Literal, Optional, TypeAlias, TypedDict, Union

from openai.types.chat import ChatCompletionMessageParam


class InferenceEngineCredentials(TypedDict):
    """Contains the prediction results and metadata for the inference.

    Args:
        api_url (str): API url of the hosted LLM server.

        api_key (str, optional) : API key to access services on the hosted LLM service.

        space_id (str, optional) : space id to use for the WML platform.

        project_id (str, optional) : project id to use for the WML platform.
    """

    api_url: str
    api_key: Optional[str] = None  # Optional for vLLM/Ollama server mode
    space_id: Optional[str] = None  # only used in WML engine
    project_id: Optional[str] = None  # only used in WML engine


class RITSInferenceEngineParams(TypedDict):
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    max_tokens: Optional[int] = 100
    seed: Optional[int] = None
    stop: Union[Optional[str], List[str]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_logprobs: Optional[int] = 20
    logit_bias: Optional[Dict[str, int]] = None
    logprobs: Optional[bool] = True
    n: Optional[int] = None
    parallel_tool_calls: Optional[bool] = None
    service_tier: Optional[Literal["auto", "default"]] = None


class WMLInferenceEngineParams(TypedDict):

    # generation params
    decoding_method: Optional[Literal["greedy", "sample"]] = None
    length_penalty: Optional[Dict[str, Union[int, float]]] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    random_seed: Optional[int] = None
    repetition_penalty: Optional[float] = None
    min_new_tokens: Optional[int] = None
    max_new_tokens: Optional[int] = 100
    stop_sequences: Optional[List[str]] = None
    time_limit: Optional[int] = None
    truncate_input_tokens: Optional[int] = None
    prompt_variables: Optional[Dict[str, Any]] = None
    return_options: Optional[Dict[str, bool]] = None

    # chat params
    frequency_penalty: float | None = None
    logprobs: bool | None = None
    top_logprobs: int | None = None
    presence_penalty: float | None = None
    response_format: dict | None = None
    temperature: float | None = None
    max_tokens: int | None = None
    time_limit: int | None = None
    top_p: float | None = None
    n: int | None = None
    logit_bias: dict | None = None
    seed: int | None = None
    stop: list[str] | None = None


class OllamaInferenceEngineParams(TypedDict):
    # load time options
    numa: Optional[bool] = None
    num_ctx: Optional[int] = None
    num_batch: Optional[int] = None
    num_gpu: Optional[int] = None
    main_gpu: Optional[int] = None
    low_vram: Optional[bool] = None
    f16_kv: Optional[bool] = None
    logits_all: Optional[bool] = None
    vocab_only: Optional[bool] = None
    use_mmap: Optional[bool] = None
    use_mlock: Optional[bool] = None
    embedding_only: Optional[bool] = None
    num_thread: Optional[int] = None

    # runtime options
    num_keep: Optional[int] = None
    seed: Optional[int] = None
    num_predict: Optional[int] = 100
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    tfs_z: Optional[float] = None
    typical_p: Optional[float] = None
    repeat_last_n: Optional[int] = None
    temperature: Optional[float] = None
    repeat_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    mirostat: Optional[int] = None
    mirostat_tau: Optional[float] = None
    mirostat_eta: Optional[float] = None
    penalize_newline: Optional[bool] = None
    stop: Optional[List[str]] = None


class VLLMInferenceEngineParams(TypedDict):

    n: int = 1
    best_of: Optional[int] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    seed: Optional[int] = None
    stop: Optional[Union[str, List[str]]] = None
    stop_token_ids: Optional[List[int]] = None
    bad_words: Optional[List[str]] = None
    ignore_eos: bool = False
    max_tokens: Optional[int] = 100
    min_tokens: int = 0
    logprobs: Optional[int] = None
    prompt_logprobs: Optional[int] = None


@dataclasses.dataclass(kw_only=True)
class TextGenerationInferenceOutput:
    """Contains the prediction results and metadata for the inference.

    Args:
        prediction (Any): model ouput

        input_tokens (int) : number of input tokens to the model.

        output_tokens (int) : number of output tokens to the model.

        stop_reason (str): stop reason for text generation, for example "eos" (end of string).

        seed (int): seed used by the model during generation.

        input_text (str): input to the model.

        model_name (str): the model_name as kept in the InferenceEngine.

        inference_engine (str): The label stating the type of the InferenceEngine.
    """

    prediction: Union[str, List[Dict[str, Any]]]
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    stop_reason: Optional[str] = None
    seed: Optional[int] = None
    input_text: Optional[str] = None
    model_name_or_path: Optional[str] = None
    inference_engine: Optional[str] = None


OpenAIChatCompletionMessageParam: TypeAlias = List[ChatCompletionMessageParam]
