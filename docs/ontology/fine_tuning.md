

# Slot: fine_tuning


_A description of the fine-tuning mechanism(s) applied to a model._





URI: [nexus:fine_tuning](https://ibm.github.io/risk-atlas-nexus/ontology/fine_tuning)
Alias: fine_tuning

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |







## Properties

* Range: [String](String.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:fine_tuning |
| native | nexus:fine_tuning |




## LinkML Source

<details>
```yaml
name: fine_tuning
description: A description of the fine-tuning mechanism(s) applied to a model.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: fine_tuning
domain_of:
- LargeLanguageModel
range: string

```
</details>
