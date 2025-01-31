

# Slot: hasEvaluation


_A relationship indicating that an entity has an AI evaluation result._





URI: [dqv:hasQualityMeasurement](dqv:hasQualityMeasurement)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |







## Properties

* Range: [AiEvalResult](AiEvalResult.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | dqv:hasQualityMeasurement |
| native | nexus:hasEvaluation |




## LinkML Source

<details>
```yaml
name: hasEvaluation
description: A relationship indicating that an entity has an AI evaluation result.
from_schema: http://research.ibm.com/ontologies/aiont/ai-risk-ontology
rank: 1000
slot_uri: dqv:hasQualityMeasurement
alias: hasEvaluation
domain_of:
- AiModel
range: AiEvalResult
multivalued: true
inlined: true
inlined_as_list: true

```
</details>