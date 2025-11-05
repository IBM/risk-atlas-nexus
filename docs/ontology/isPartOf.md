

# Slot: isPartOf


_A relationship where an entity is part of another entity_





URI: [schema:isPartOf](http://schema.org/isPartOf)
Alias: isPartOf

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Risk](Risk.md) | The state of uncertainty associated with an AI system, that has the potential... |  yes  |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  yes  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |
| [Stakeholder](Stakeholder.md) | An AI system stakeholder for Responsible AI governance (e |  yes  |







## Properties

* Range: [String](String.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:isPartOf |
| native | nexus:isPartOf |




## LinkML Source

<details>
```yaml
name: isPartOf
description: A relationship where an entity is part of another entity
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: schema:isPartOf
alias: isPartOf
domain_of:
- Risk
- LargeLanguageModel
- Stakeholder
range: string

```
</details>
