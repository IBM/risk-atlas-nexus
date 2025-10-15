

# Slot: numTrainingTokens


_The number of tokens a AI model was trained on._





URI: [nexus:numTrainingTokens](https://ibm.github.io/risk-atlas-nexus/ontology/numTrainingTokens)
Alias: numTrainingTokens

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
| self | nexus:numTrainingTokens |
| native | nexus:numTrainingTokens |




## LinkML Source

<details>
```yaml
name: numTrainingTokens
description: The number of tokens a AI model was trained on.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: numTrainingTokens
domain_of:
- LargeLanguageModel
range: integer
minimum_value: 0

```
</details>
