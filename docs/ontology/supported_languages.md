

# Slot: supported_languages


_A list of languages, expressed as ISO two letter codes. For example, 'jp, fr, en, de'_





URI: [nexus:supported_languages](https://ibm.github.io/risk-atlas-nexus/ontology/supported_languages)
Alias: supported_languages

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |







## Properties

* Range: [String](String.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:supported_languages |
| native | nexus:supported_languages |




## LinkML Source

<details>
```yaml
name: supported_languages
description: A list of languages, expressed as ISO two letter codes. For example,
  'jp, fr, en, de'
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
alias: supported_languages
domain_of:
- LargeLanguageModel
range: string
multivalued: true
inlined: true
inlined_as_list: true

```
</details>
