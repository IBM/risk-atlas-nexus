

# Slot: isResultOf


_A relationship indicating that an entity is the result of an AI evaluation._





URI: [dqv:isMeasurementOf](https://www.w3.org/TR/vocab-dqv/isMeasurementOf)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [AiEvalResult](AiEvalResult.md) | The result of an evaluation for a specific AI model |  no  |







## Properties

* Range: [AiEval](AiEval.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | dqv:isMeasurementOf |
| native | nexus:isResultOf |




## LinkML Source

<details>
```yaml
name: isResultOf
description: A relationship indicating that an entity is the result of an AI evaluation.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: dqv:isMeasurementOf
alias: isResultOf
domain_of:
- AiEvalResult
range: AiEval
inlined: false

```
</details>