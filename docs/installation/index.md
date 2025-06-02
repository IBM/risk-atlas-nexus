This project targets python version ">=3.11, <3.12". You can download specific versions of python here: [https://www.python.org/downloads/](https://www.python.org/downloads/)

Install the risk_atlas_nexus library
```
git clone git@github.com:IBM/risk-atlas-nexus.git
cd risk-atlas-nexus
python -m venv vrisk-atlas-nexus
source vrisk-atlas-nexus/bin/activate
pip install -e .
```

### Install for inference APIs

Risk Atlas Nexus uses Large Language Models (LLMs) to infer risks and risks data. Therefore, requires access to LLMs to inference or call the model. The following LLM inference APIs are supported:

- [IBM Watsonx AI](https://www.ibm.com/products/watsonx-ai) (Watson Machine Learning)
- [Ollama](https://ollama.com/)
- [vLLM](https://docs.vllm.ai/en/latest/)
- [RITS](https://rits.fmaas.res.ibm.com) (IBM Internal Only)

#### IBM Watsonx AI (WML)

When using the WML platform, you need to:

1. Add configuration to `.env` file as follows. Please follow this [documentation](https://dataplatform.cloud.ibm.com/docs/content/wsj/analyze-data/fm-credentials.html?context=wx&locale=en) on obtaining WML credentials.

```yaml
WML_API_KEY=<WML api key goes here>
WML_API_URL=<WML url key goes here>
WML_PROJECT_ID=<WML project id goes here, Optional>
WML_SPACE_ID=<WML space id goes here, Optional>
```

Either 'WML_PROJECT_ID' or 'WML_SPACE_ID' need to be specified.

2. Install WML dependencies as follows:

```command
pip install -e ".[wml]"
```

#### Ollama

When using the Ollama inference, you need to:

1. Install Ollama dependencies as follows:

```command
pip install -e ".[ollama]"
```

2. Please follow the [quickstart](https://github.com/ollama/ollama/blob/main/README.md#ollama) guide to start Ollama LLM server. Server will start by default at http://localhost:11434

3. When selecting Ollama engine in Risk Atlas Nexus, use the server address `localhost:11434` as the `api_url` in the credentials or set the environment variable `OLLAMA_API_URL` with this value.

#### vLLM

When using the vLLM inference, you need to:

1. For Mac users, follow the instuctions [here](https://docs.vllm.ai/en/stable/getting_started/installation/cpu/index.html?device=apple). Users need to build from the source vLLM to natively run on macOS.

2. For Linux users, install vLLM dependencies as follows:

```command
pip install -e ".[vllm]"
```

Above package is enough to run vLLM in once-off offline mode. When selecting vLLM execution from Risk Atlas Nexus, `credentials` should be passed as `None` to use vLLM offline mode.

3. (Optional) To run vLLM on an OpenAI-Compatible vLLM Server, execute the command:

```command
vllm serve ibm-granite/granite-3.1-8b-instruct --max_model_len 4096 --host localhost --port 8000 --api-key <CUSTOM_API_KEY>
```

The CUSTOM_API_KEY can be any string that you choose to use as your API key. Above command will start vLLM server at http://localhost:8000. The server currently hosts one model at a time. Check all supported APIs at `http://localhost:8000/docs`

**Note:** When selecting vLLM engine in Risk Atlas Nexus, pass `api_url` as `host:port` and given `api_key` to `credentials` with values from the vllm serve command above.

#### RITS (IBM Internal Only)

When using the RITS platform, you need to:

1. Add configuration to `.env` file as follows:

```yaml
RITS_API_KEY=<RITS api key goes here>
RITS_API_URL=<RITS url key goes here>
```

2. Install RITS dependencies as follows:

```command
pip install -e ".[rits]"
```
