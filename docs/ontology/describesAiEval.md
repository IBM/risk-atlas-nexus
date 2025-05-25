

# Slot: describesAiEval


_A relationship where a BenchmarkMetadataCard describes and AI evaluation (benchmark)._





URI: [nexus:describesAiEval](https://ibm.github.io/risk-atlas-nexus/ontology/describesAiEval)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | Benchmark metadata cards offer a standardized way to document LLM benchmarks ... |  no  |







## Properties

* Range: [AiEval](AiEval.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:describesAiEval |
| native | nexus:describesAiEval |




## LinkML Source

<details>
```yaml
name: describesAiEval
description: A relationship where a BenchmarkMetadataCard describes and AI evaluation
  (benchmark).
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
domain: BenchmarkMetadataCard
alias: describesAiEval
domain_of:
- BenchmarkMetadataCard
inverse: hasBenchmarkMetadata
range: AiEval
multivalued: true
inlined: false

```
</details>