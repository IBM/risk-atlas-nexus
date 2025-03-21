

# Slot: hasPart


_A relationship where an entity has another entity_





URI: [schema:hasPart](http://schema.org/hasPart)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RiskGroup](RiskGroup.md) | A group of AI system related risks that are part of a risk taxonomy |  yes  |







## Properties

* Range: [String](String.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:hasPart |
| native | nexus:hasPart |




## LinkML Source

<details>
```yaml
name: hasPart
description: A relationship where an entity has another entity
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: schema:hasPart
alias: hasPart
domain_of:
- RiskGroup
range: string
multivalued: true

```
</details>