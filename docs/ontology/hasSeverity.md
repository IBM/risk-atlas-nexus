

# Slot: hasSeverity


_Indicates the severity associated with a concept_





URI: [nexus:hasSeverity](https://ibm.github.io/risk-atlas-nexus/ontology/hasSeverity)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RiskIncident](RiskIncident.md) | An event occuring or occured which is a realised or materialised risk |  no  |







## Properties

* Range: [Severity](Severity.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:hasSeverity |
| native | nexus:hasSeverity |




## LinkML Source

<details>
```yaml
name: hasSeverity
description: Indicates the severity associated with a concept
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
domain: RiskConcept
alias: hasSeverity
domain_of:
- RiskIncident
range: Severity

```
</details>