

# Slot: hasRelatedRisk


_A relationship where an entity relates to a risk_





URI: [nexus:hasRelatedRisk](https://ibm.github.io/risk-atlas-nexus/ontology/hasRelatedRisk)
Alias: hasRelatedRisk

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |
| [Term](Term.md) | A term and its definitions |  no  |
| [LLMIntrinsic](LLMIntrinsic.md) | A capability that can be invoked through a well-defined API that is reasonabl... |  no  |
| [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | Benchmark metadata cards offer a standardized way to document LLM benchmarks ... |  no  |
| [Action](Action.md) | Action to remediate a risk |  no  |
| [Adapter](Adapter.md) | Adapter-based methods add extra trainable parameters after the attention and ... |  no  |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [LLMQuestionPolicy](LLMQuestionPolicy.md) | The policy guides how the language model should answer a diverse set of sensi... |  no  |







## Properties

* Range: [Risk](Risk.md)&nbsp;or&nbsp;<br />[RiskConcept](RiskConcept.md)&nbsp;or&nbsp;<br />[Term](Term.md)

* Multivalued: True





## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:hasRelatedRisk |
| native | nexus:hasRelatedRisk |




## LinkML Source

<details>
```yaml
name: hasRelatedRisk
description: A relationship where an entity relates to a risk
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
domain: Any
alias: hasRelatedRisk
domain_of:
- Term
- LLMQuestionPolicy
- Action
- AiEval
- BenchmarkMetadataCard
- Adapter
- LLMIntrinsic
range: Risk
multivalued: true
inlined: false
any_of:
- range: RiskConcept
- range: Term

```
</details>
