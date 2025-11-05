

# Slot: evidence


_Evidence provides a source (typical a chunk, paragraph or link) describing where some value was found or how it was generated._





URI: [nexus:evidence](https://ibm.github.io/risk-atlas-nexus/ontology/evidence)
Alias: evidence

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Fact](Fact.md) | A fact about something, for example the result of a measurement |  no  |
| [AiEvalResult](AiEvalResult.md) | The result of an evaluation for a specific AI model |  no  |







## Properties

* Range: [String](String.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:evidence |
| native | nexus:evidence |




## LinkML Source

<details>
```yaml
name: evidence
description: Evidence provides a source (typical a chunk, paragraph or link) describing
  where some value was found or how it was generated.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: evidence
domain_of:
- Fact
range: string

```
</details>
