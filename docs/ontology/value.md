

# Slot: value


_Some numeric or string value_





URI: [nexus:value](https://ibm.github.io/risk-atlas-nexus/ontology/value)
Alias: value

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [AiEvalResult](AiEvalResult.md) | The result of an evaluation for a specific AI model |  no  |
| [Fact](Fact.md) | A fact about something, for example the result of a measurement |  no  |







## Properties

* Range: [String](String.md)

* Required: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:value |
| native | nexus:value |




## LinkML Source

<details>
```yaml
name: value
description: Some numeric or string value
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: value
domain_of:
- Fact
range: string
required: true

```
</details>
