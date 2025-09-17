

# Slot: version


_The version of the entity embodied by a specified resource._





URI: [schema:version](http://schema.org/version)
Alias: version

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Vocabulary](Vocabulary.md) | A collection of terms, with their definitions and relationships |  no  |
| [License](License.md) | The general notion of a license which defines terms and grants permissions to... |  no  |
| [RiskTaxonomy](RiskTaxonomy.md) | A taxonomy of AI system related risks |  no  |







## Properties

* Range: [String](String.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:version |
| native | nexus:version |




## LinkML Source

<details>
```yaml
name: version
description: The version of the entity embodied by a specified resource.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: schema:version
alias: version
domain_of:
- License
- Vocabulary
- RiskTaxonomy
range: string

```
</details>
