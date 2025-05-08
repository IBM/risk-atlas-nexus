

# Slot: hasDataSource


_The origin or source of the data used in the benchmark (e.g., curated datasets, user submissions)._





URI: [nexus:hasDataSource](https://ibm.github.io/risk-atlas-nexus/ontology/hasDataSource)



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
| self | nexus:hasDataSource |
| native | nexus:hasDataSource |




## LinkML Source

<details>
```yaml
name: hasDataSource
description: The origin or source of the data used in the benchmark (e.g., curated
  datasets, user submissions).
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: hasDataSource
domain_of:
- BenchmarkMetadataCard
range: string
multivalued: true

```
</details>