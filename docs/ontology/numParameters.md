

# Slot: numParameters


_A property indicating the number of parameters in a LLM._





URI: [nexus:numParameters](https://ibm.github.io/risk-atlas-nexus/ontology/numParameters)
Alias: numParameters

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |







## Properties

* Range: [Integer](Integer.md)

* Minimum Value: 0





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:numParameters |
| native | nexus:numParameters |




## LinkML Source

<details>
```yaml
name: numParameters
description: A property indicating the number of parameters in a LLM.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: numParameters
domain_of:
- LargeLanguageModel
range: integer
minimum_value: 0

```
</details>
