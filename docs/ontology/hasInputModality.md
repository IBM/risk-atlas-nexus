

# Slot: hasInputModality


_A relationship indicating the input modalities supported by an AI component. Examples include text, image, video._





URI: [nexus:hasInputModality](https://ibm.github.io/risk-atlas-nexus/ontology/hasInputModality)
Alias: hasInputModality

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |







## Properties

* Range: [Modality](Modality.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:hasInputModality |
| native | nexus:hasInputModality |




## LinkML Source

<details>
```yaml
name: hasInputModality
description: A relationship indicating the input modalities supported by an AI component.
  Examples include text, image, video.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: hasInputModality
domain_of:
- LargeLanguageModel
range: Modality
multivalued: true
inlined: false

```
</details>
