

# Slot: isDefinedByTaxonomy


_A relationship where a risk or a risk group is defined by a risk taxonomy_





URI: [schema:isPartOf](http://schema.org/isPartOf)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [RiskControl](RiskControl.md) | A measure that maintains and/or modifies risk (and risk concepts) |  no  |
| [Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential... |  no  |
| [RiskGroup](RiskGroup.md) | A group of AI system related risks that are part of a risk taxonomy |  no  |
| [Action](Action.md) | Action to remediate a risk |  no  |







## Properties

* Range: [RiskTaxonomy](RiskTaxonomy.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:isPartOf |
| native | nexus:isDefinedByTaxonomy |




## LinkML Source

<details>
```yaml
name: isDefinedByTaxonomy
description: A relationship where a risk or a risk group is defined by a risk taxonomy
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: schema:isPartOf
alias: isDefinedByTaxonomy
domain_of:
- RiskGroup
- Risk
- RiskControl
- Action
range: RiskTaxonomy

```
</details>