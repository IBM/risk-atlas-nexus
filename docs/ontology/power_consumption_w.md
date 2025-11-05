

# Slot: power_consumption_w


_power consumption in Watts_





URI: [nexus:power_consumption_w](https://ibm.github.io/risk-atlas-nexus/ontology/power_consumption_w)
Alias: power_consumption_w

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
| self | nexus:power_consumption_w |
| native | nexus:power_consumption_w |




## LinkML Source

<details>
```yaml
name: power_consumption_w
description: power consumption in Watts
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: power_consumption_w
domain_of:
- AiModel
range: integer
minimum_value: 0

```
</details>
