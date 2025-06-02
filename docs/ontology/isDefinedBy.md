

# Slot: isDefinedBy


_A relationship where a risk or a risk group is defined by a risk taxonomy_





URI: [schema:isPartOf](http://schema.org/isPartOf)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential... |  no  |
| [RiskGroup](RiskGroup.md) | A group of AI system related risks that are part of a risk taxonomy |  no  |







## Properties

* Range: [RiskTaxonomy](RiskTaxonomy.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://www.ibm.com/docs/en/watsonx/saas?topic=ontology-ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:isPartOf |
| native | nexus:isDefinedBy |




## LinkML Source

<details>
```yaml
name: isDefinedBy
description: A relationship where a risk or a risk group is defined by a risk taxonomy
from_schema: https://www.ibm.com/docs/en/watsonx/saas?topic=ontology-ai-risk-ontology
rank: 1000
slot_uri: schema:isPartOf
alias: isDefinedBy
domain_of:
- RiskGroup
- Risk
range: RiskTaxonomy

```
</details>
