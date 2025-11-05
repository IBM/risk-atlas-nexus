

# Slot: gpu_hours


_GPU consumption in terms of hours_





URI: [nexus:gpu_hours](https://ibm.github.io/risk-atlas-nexus/ontology/gpu_hours)
Alias: gpu_hours

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |







## Properties

* Range: [Integer](Integer.md)

* Minimum Value: 0





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:gpu_hours |
| native | nexus:gpu_hours |




## LinkML Source

<details>
```yaml
name: gpu_hours
description: GPU consumption in terms of hours
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: gpu_hours
domain_of:
- AiModel
range: integer
minimum_value: 0

```
</details>
