

# Slot: hasDataset


_A relationship to datasets that are used._





URI: [nexus:hasDataset](https://ibm.github.io/risk-atlas-nexus/ontology/hasDataset)
Alias: hasDataset

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |







## Properties

* Range: [Dataset](Dataset.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:hasDataset |
| native | nexus:hasDataset |




## LinkML Source

<details>
```yaml
name: hasDataset
description: A relationship to datasets that are used.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: hasDataset
domain_of:
- AiEval
range: Dataset
multivalued: true
inlined: false

```
</details>
