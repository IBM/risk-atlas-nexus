

# Slot: aimodels


_A list of AI models_





URI: [nexus:aimodels](https://ibm.github.io/risk-atlas-nexus/ontology/aimodels)



<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Container](Container.md) | An umbrella object that holds the ontology class instances |  no  |







## Properties

* Range: [LargeLanguageModel](LargeLanguageModel.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:aimodels |
| native | nexus:aimodels |




## LinkML Source

<details>
```yaml
name: aimodels
description: A list of AI models
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: aimodels
owner: Container
domain_of:
- Container
range: LargeLanguageModel
multivalued: true
inlined: true
inlined_as_list: true

```
</details>