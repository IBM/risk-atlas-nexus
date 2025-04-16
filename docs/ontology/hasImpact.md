

# Slot: hasImpact


_Indicates impact(s) possible or arising as consequences from specified concept_





URI: [nexus:hasImpact](https://ibm.github.io/risk-atlas-nexus/ontology/hasImpact)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RiskIncident](RiskIncident.md) | An event occuring or occured which is a realised or materialised risk |  no  |







## Properties

* Range: [Impact](Impact.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:hasImpact |
| native | nexus:hasImpact |
| broad | dpv:hasConsequence |




## LinkML Source

<details>
```yaml
name: hasImpact
description: Indicates impact(s) possible or arising as consequences from specified
  concept
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
broad_mappings:
- dpv:hasConsequence
rank: 1000
domain: RiskConcept
alias: hasImpact
domain_of:
- RiskIncident
range: Impact

```
</details>