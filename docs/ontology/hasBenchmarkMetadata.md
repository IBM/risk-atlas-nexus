

# Slot: hasBenchmarkMetadata


_A relationship to a Benchmark Metadata Card which contains metadata about the benchmark._





URI: [nexus:hasBenchmarkMetadata](https://ibm.github.io/risk-atlas-nexus/ontology/hasBenchmarkMetadata)
Alias: hasBenchmarkMetadata

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |







## Properties

* Range: [BenchmarkMetadataCard](BenchmarkMetadataCard.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:hasBenchmarkMetadata |
| native | nexus:hasBenchmarkMetadata |




## LinkML Source

<details>
```yaml
name: hasBenchmarkMetadata
description: A relationship to a Benchmark Metadata Card which contains metadata about
  the benchmark.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
domain: AiEval
alias: hasBenchmarkMetadata
domain_of:
- AiEval
inverse: describesAiEval
range: BenchmarkMetadataCard
multivalued: true
inlined: false

```
</details>
