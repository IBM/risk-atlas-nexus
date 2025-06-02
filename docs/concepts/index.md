The [IBM AI Risk Atlas](IBM_AI_Risk_Atlas.md) has been used many of IBM's enterprise customers to help them reason about the risks
in their AI systems. In order to enable efforts to leverage these risks to operationalise governance and
risk mitigation frameworks we created Risk Atlas Nexus.

The Risk Atlas Nexus is a collection of tooling and resources related to governance of foundation models. It contributes an [ontology](../ontology/index.md) and a knowledge graph which contains the risks from different existing taxonomies (for example; the OWASP Top 10 for LLMs and Generative AI Apps, the NIST AI Risk Management Framework, the MIT AI Risk Repository, the AILuminate Benchmark, the AIR taxonomy 2024), Credo's Unified Control Framework) and the mappings between them.

The ontology has been modeled using [LinkML](https://linkml.io/linkml/index.html), which allows the generation of different data representations (e.g. RDF, OWL) in a simple way. The risk taxonomies have been stored as LinkML data instance YAML files.

A [python implementation](../reference/index.md) is provided to traverse the underlying graph, to more easily allow users to make use of the abstract risk definitions into their own actionable workflows and AI governance processes.

Compliance questionnaires are usually required prior to deploying an AI model into production. These enable a thorough understanding of the specific use case and associated risk exposures. The Risk Atlas Nexus supports the development and curation of questionnaires to a desired taxonomy. Additionally, the content can support Large Language Models (LLMs) to assist users in responding to time-consuming compliance questionnaires, thereby reducing manual effort and
minimizing errors
