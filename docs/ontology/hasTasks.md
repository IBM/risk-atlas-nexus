

# Slot: hasTasks


_The tasks or evaluations the benchmark is intended to assess._





URI: [nexus:hasTasks](https://ibm.github.io/risk-atlas-nexus/ontology/hasTasks)
Alias: hasTasks

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | Benchmark metadata cards offer a standardized way to document LLM benchmarks ... |  no  |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |







## Properties

* Range: [String](String.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:hasTasks |
| native | nexus:hasTasks |




## LinkML Source

<details>
```yaml
name: hasTasks
description: The tasks or evaluations the benchmark is intended to assess.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: hasTasks
domain_of:
- AiEval
- BenchmarkMetadataCard
range: string
multivalued: true
inlined: false

```
</details>
