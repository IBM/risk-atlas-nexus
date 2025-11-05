

# Slot: architecture


_A description of the architecture of an AI such as 'Decoder-only'._





URI: [nexus:architecture](https://ibm.github.io/risk-atlas-nexus/ontology/architecture)
Alias: architecture

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |







## Properties

* Range: [String](String.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:architecture |
| native | nexus:architecture |




## LinkML Source

<details>
```yaml
name: architecture
description: A description of the architecture of an AI such as 'Decoder-only'.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: architecture
domain_of:
- AiModel
range: string

```
</details>
