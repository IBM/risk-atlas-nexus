

# Slot: hasRelatedRisk


_A relationship where an entity relates to a risk_





URI: [nexus:hasRelatedRisk](https://ibm.github.io/risk-atlas-nexus/ontology/hasRelatedRisk)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |
| [Action](Action.md) | Action to remediate a risk |  no  |







## Properties

* Range: [Risk](Risk.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:hasRelatedRisk |
| native | nexus:hasRelatedRisk |




## LinkML Source

<details>
```yaml
name: hasRelatedRisk
description: A relationship where an entity relates to a risk
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
domain: RiskConcept
alias: hasRelatedRisk
domain_of:
- Action
- AiEval
range: Risk
multivalued: true
inlined: false

```
</details>
