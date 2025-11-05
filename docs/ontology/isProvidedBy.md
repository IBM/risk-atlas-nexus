

# Slot: isProvidedBy


_A relationship indicating the AI model has been provided by an AI model provider._





URI: [airo:isProvidedBy](https://w3id.org/airo#isProvidedBy)
Alias: isProvidedBy

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [BaseAi](BaseAi.md) | Any type of AI, be it a LLM, RL agent, SVM, etc |  no  |
| [AiSystem](AiSystem.md) | A compound AI System composed of one or more AI capablities |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [AiAgent](AiAgent.md) | An artificial intelligence (AI) agent refers to a system or program that is c... |  yes  |







## Properties

* Range: [AiProvider](AiProvider.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | airo:isProvidedBy |
| native | nexus:isProvidedBy |




## LinkML Source

<details>
```yaml
name: isProvidedBy
description: A relationship indicating the AI model has been provided by an AI model
  provider.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: airo:isProvidedBy
alias: isProvidedBy
domain_of:
- BaseAi
range: AiProvider

```
</details>
