

# Slot: isDefinedByVocabulary


_A relationship where a term or a term group is defined by a vocabulary_





URI: [schema:isPartOf](http://schema.org/isPartOf)
Alias: isDefinedByVocabulary

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [Term](Term.md) | A term and its definitions |  no  |
| [LLMIntrinsic](LLMIntrinsic.md) | A capability that can be invoked through a well-defined API that is reasonabl... |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |







## Properties

* Range: [Vocabulary](Vocabulary.md)





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:isPartOf |
| native | nexus:isDefinedByVocabulary |




## LinkML Source

<details>
```yaml
name: isDefinedByVocabulary
description: A relationship where a term or a term group is defined by a vocabulary
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: schema:isPartOf
alias: isDefinedByVocabulary
domain_of:
- Term
- Adapter
- LLMIntrinsic
range: Vocabulary

```
</details>
