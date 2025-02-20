# Risk Atlas Nexus

<img src="resources/images/risk_atlas_nexus_vector.svg" width="200">

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0) ![main branch](https://github.com/IBM/risk-atlas-nexus/actions/workflows/pages/pages-build-deployment/badge.svg?branch=main) [![](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

## Overview
Risk Atlas Nexus aims to provide tooling to help bring together disparate resources related to governance of foundation models. We support a community driven approach to curating and cataloguing resources such as datasets, benchmarks and mitigations.  Our goal is to turn abstract risk definitions into actionable workflows that streamline AI governance processes. By connecting fragmented resources, Risk Atlas Nexus seeks to fill a critical gap in AI governance, enabling stakeholders to build more robust, transparent, and accountable systems. Risk Atlas Nexus builds on the [IBM AI Risk Atlas](https://www.ibm.com/docs/en/watsonx/saas?topic=ai-risk-atlas) making this educational resource a nexus of governance assets and tooling.  An AI System's Knowledge Graph is used to provide a unified structure that links and contextualize the very heterogeneous domain data. 

Our intention is to create a starting point for an open AI Systems ontology whose focus is on risk and that the community can extend and enhance. This ontology serves as the foundation that unifies innovation and tooling in the AI risk space. By lowering the barrier to entry for developers, it fosters a governance-first approach to AI solutions, while also inviting the broader community to contribute their own tools and methodologies to expand its impact.

## Features
- 🏗️📊 An ontology has been provided, that combines the AI risk view (taxonomies, risks, actions) with an AI model view (AI systems, AI models, model evaluations) into one coherent schema 
- 📚⚠️ AI Risks were collected from IBM AI Risk Atlas, IBM Granite Guardian, MIT AI Risk Repository, NIST Artificial Intelligence Risk Management Framework: Generative Artificial Intelligence Profile, and OWASP Top 10 for Large Language Model Applications
- 🔗📌 Mappings are proposed between the taxonomies and between risks and actions
- 🐍🔍 Use the python library methods to quickly explore available risks, relations and actions 
- 🚨🧐 Use the python library methods to detect potential risks in your usecase 
- 📤📈 Download an exported graph populated with data instances

## Architecture
![Architecture](resources/images/architecture.png)

### Coming soon
- Tooling to convert the LinkML schema and instance data into a Cypher representation to populate a graph database
- Example use-case of auto-assistance in compliance questionnaires using CoT examples and Risk Atlas Nexus


## Links
- **AI Risk Ontology**
    - [LinkML schema documentation](docs/ontology/index.md)
    - [LinkML instance data for an example knowledge graph](src/risk_atlas_nexus/data/knowledge_graph/README.md)
    - [Download a populated graph](graph_export/README.md)
- **Notebooks:** 
    - [Risk identification](examples/notebooks/risk_identification.ipynb) Uncover risks related to your usecase
    - [Risk Atlas Nexus Quickstart](examples/notebooks/Risk Atlas Nexus Quickstart.ipynb) Overview of library functionality
    - [Risk identification](examples/notebooks/risk_identification.ipynb) Uncover risks related to your usecase
    - [Risk Atlas Nexus Quickstart](examples/notebooks/Risk Atlas Nexus Quickstart.ipynb) Overview of library functionality
- **Additional Resources:**
    - [IBM AI Risk Atlas](https://www.ibm.com/docs/en/watsonx/saas?topic=ai-risk-atlas)
    - [Usage Governance Advisor: From Intent to AI Governance](https://arxiv.org/abs/2412.01957)


## Installation
This project targets python version ">=3.11, <3.12". You can download specific versions of python here: https://www.python.org/downloads/

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

- [IBM Watsonx AI (WML)](https://www.ibm.com/products/watsonx-ai)
- [Ollama](https://ollama.com/)
- [vLLM](https://docs.vllm.ai/en/latest/)
- [RITS](https://rits.fmaas.res.ibm.com)

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

2. Please follow the [docs](https://github.com/ollama/ollama/tree/main/docs) here to start Ollama LLM server. Server will start by default at http://localhost:11434

3. When selecting Ollama execution from Risk Atlas Nexus, pass the `api_url` to the `credentials` or set `OLLAMA_API_URL` with the value as `localhost:11434`.

#### vLLM 

When using the vLLM inference, you need to:

1. Install vLLM dependencies as follows:

```command
pip install -e ".[vllm]"
```

Above package is enough to run vLLM in once-off offline mode. When selecting vLLM execution from Risk Atlas Nexus, `credentials` should be passed as `None` to use vLLM offline mode.

2. (Optional) To run vLLM on an OpenAI-Compatible vLLM Server, execute the command:

```command
vllm serve ibm-granite/granite-3.1-8b-instruct --max_model_len 4096 --host localhost --port 8000 --api-key <CUSTOM_API_KEY>
```

The CUSTOM_API_KEY can be any string that you choose to use as your API key. Above command will start vLLM server at http://localhost:8000. The server currently hosts one model at a time. Check all supported APIs at `http://localhost:8000/docs`

**Note:** When selecting vLLM execution from Risk Atlas Nexus, pass `api_url (host:port)` and `api_key` to `credentials` with values from the vllm serve command above.

#### RITS 

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

## Compatibility
- View the [changelog](CHANGELOG.md).

## Contributing
- Get started by checking our [contribution guidelines](CONTRIBUTING.md).
- Read the wiki for more technical and design details.
- If you have any questions, just ask!

Tip: Use the makefile provided to regenerate artifacts provided in the repository by running `make` in this repository.

## Find out more
- [IBM's AI Risk Atlas](https://www.ibm.com/docs/en/watsonx/saas?topic=ai-risk-atlas)
- Read the the IBM AI Ethics Board publication [Foundation models: Opportunities, risks and mitigations](https://www.ibm.com/downloads/documents/us-en/10a99803d8afd656) which goes into more detail about the risk taxonomy, and describes the point of view of IBM on the ethics of foundation models.
- ['Usage Governance Advisor: From Intent to AI Governance'](https://arxiv.org/abs/2412.01957) presents a system for semi-structured governance information, identifying and prioritising risks according to the intended use case, recommending appropriate benchmarks and risk assessments and proposing mitigation strategies and actions.

## IBM ❤️ Open Source AI

Risk Atlas Nexus has been brought to you by IBM.

