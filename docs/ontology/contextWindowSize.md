

# Slot: contextWindowSize


_The total length, in bytes, of an AI model's context window._





URI: [nexus:contextWindowSize](https://ibm.github.io/risk-atlas-nexus/ontology/contextWindowSize)
Alias: contextWindowSize

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
| self | nexus:contextWindowSize |
| native | nexus:contextWindowSize |




## LinkML Source

<details>
```yaml
name: contextWindowSize
description: The total length, in bytes, of an AI model's context window.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: contextWindowSize
domain_of:
- LargeLanguageModel
range: integer
minimum_value: 0

```
</details>
