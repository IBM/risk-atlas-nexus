

# Slot: refersToRisk


_Indicates the incident (subject) is a materialisation of the indicated risk (object)_





URI: [nexus:refersToRisk](https://ibm.github.io/risk-atlas-nexus/ontology/refersToRisk)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RiskIncident](RiskIncident.md) | An event occuring or occured which is a realised or materialised risk |  no  |







## Properties

* Range: [Risk](Risk.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:refersToRisk |
| native | nexus:refersToRisk |
| exact | dpv:refersToRisk |




## LinkML Source

<details>
```yaml
name: refersToRisk
description: Indicates the incident (subject) is a materialisation of the indicated
  risk (object)
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
exact_mappings:
- dpv:refersToRisk
rank: 1000
domain: RiskIncident
alias: refersToRisk
domain_of:
- RiskIncident
range: Risk
multivalued: true
inlined: false

```
</details>