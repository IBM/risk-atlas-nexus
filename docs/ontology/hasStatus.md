

# Slot: hasStatus


_Indicates the status of specified concept_





URI: [nexus:hasStatus](https://ibm.github.io/risk-atlas-nexus/ontology/hasStatus)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RiskIncident](RiskIncident.md) | An event occuring or occured which is a realised or materialised risk |  no  |







## Properties

* Range: [IncidentStatus](IncidentStatus.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:hasStatus |
| native | nexus:hasStatus |




## LinkML Source

<details>
```yaml
name: hasStatus
description: Indicates the status of specified concept
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
domain: RiskConcept
alias: hasStatus
domain_of:
- RiskIncident
range: IncidentStatus

```
</details>
