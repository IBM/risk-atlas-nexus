

# Slot: hasTrainingData


_A relationship indicating the datasets an AI model was trained on._





URI: [airo:hasTrainingData](https://w3id.org/airo#hasTrainingData)
Alias: hasTrainingData

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |







## Properties

* Range: [Dataset](Dataset.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | airo:hasTrainingData |
| native | nexus:hasTrainingData |




## LinkML Source

<details>
```yaml
name: hasTrainingData
description: A relationship indicating the datasets an AI model was trained on.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: airo:hasTrainingData
alias: hasTrainingData
domain_of:
- LargeLanguageModel
range: Dataset
multivalued: true
inlined: false

```
</details>
