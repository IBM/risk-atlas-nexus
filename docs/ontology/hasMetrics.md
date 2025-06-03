

# Slot: hasMetrics


_The specific performance metrics used to assess models (e.g., accuracy, F1 score, precision, recall)._





URI: [nexus:hasMetrics](https://ibm.github.io/risk-atlas-nexus/ontology/hasMetrics)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | Benchmark metadata cards offer a standardized way to document LLM benchmarks ... |  no  |







## Properties

* Range: [String](String.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:hasMetrics |
| native | nexus:hasMetrics |




## LinkML Source

<details>
```yaml
name: hasMetrics
description: The specific performance metrics used to assess models (e.g., accuracy,
  F1 score, precision, recall).
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: hasMetrics
domain_of:
- BenchmarkMetadataCard
range: string
multivalued: true

```
</details>
