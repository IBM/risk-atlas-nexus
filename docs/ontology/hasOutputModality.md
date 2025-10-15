

# Slot: hasOutputModality


_A relationship indicating the output modalities supported by an AI component. Examples include text, image, video._





URI: [nexus:hasOutputModality](https://ibm.github.io/risk-atlas-nexus/ontology/hasOutputModality)
Alias: hasOutputModality

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
| self | nexus:hasOutputModality |
| native | nexus:hasOutputModality |




## LinkML Source

<details>
```yaml
name: hasOutputModality
description: A relationship indicating the output modalities supported by an AI component.
  Examples include text, image, video.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: hasOutputModality
domain_of:
- LargeLanguageModel
range: Modality
multivalued: true
inlined: false

```
</details>
