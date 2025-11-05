

# Slot: carbon_emitted


_The number of tons of carbon dioxide equivalent that are emitted during training_





URI: [nexus:carbon_emitted](https://ibm.github.io/risk-atlas-nexus/ontology/carbon_emitted)
Alias: carbon_emitted

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |







## Properties

* Range: [Float](Float.md)

* Minimum Value: 0





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:carbon_emitted |
| native | nexus:carbon_emitted |




## LinkML Source

<details>
```yaml
name: carbon_emitted
description: The number of tons of carbon dioxide equivalent that are emitted during
  training
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: carbon_emitted
domain_of:
- AiModel
range: float
minimum_value: 0
unit:
  symbol: t CO2-eq
  descriptive_name: tons of CO2 equivalent

```
</details>
