

# Slot: hasRiskControl


_Indicates the control measures associated with a system or component to modify risks._





URI: [airo:hasRiskControl](https://w3id.org/airo#hasRiskControl)
Alias: hasRiskControl

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LargeLanguageModel](LargeLanguageModel.md) | A large language model (LLM) is an AI model which supports a range of languag... |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |
| [AiModel](AiModel.md) | A base AI Model class |  no  |







## Properties

* Range: [RiskControl](RiskControl.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | airo:hasRiskControl |
| native | nexus:hasRiskControl |




## LinkML Source

<details>
```yaml
name: hasRiskControl
description: Indicates the control measures associated with a system or component
  to modify risks.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
slot_uri: airo:hasRiskControl
alias: hasRiskControl
domain_of:
- AiModel
range: RiskControl
multivalued: true

```
</details>
