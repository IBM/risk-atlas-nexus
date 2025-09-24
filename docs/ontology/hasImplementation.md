

# Slot: hasImplementation


_A relationship to a implementation defining the risk evaluation_





URI: [schema:url](http://schema.org/url)
Alias: hasImplementation

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |







## Properties

* Range: [Uri](Uri.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:url |
| native | nexus:hasImplementation |




## LinkML Source

<details>
```yaml
name: hasImplementation
description: A relationship to a implementation defining the risk evaluation
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: schema:url
alias: hasImplementation
domain_of:
- AiEval
range: uri
multivalued: true
inlined: false

```
</details>
