

# Slot: hasRelatedRisk


_A relationship where an entity relates to a risk_





URI: [nexus:hasRelatedRisk](https://ibm.github.io/risk-atlas-nexus/ontology/hasRelatedRisk)
Alias: hasRelatedRisk

<!-- no inheritance hierarchy -->





## Applicable Classes

| Name | Description | Modifies Slot |
| --- | --- | --- |
| [LLMIntrinsic](LLMIntrinsic.md) | A capability that can be invoked through a well-defined API that is reasonabl... |  no  |
| [AiEval](AiEval.md) | An AI Evaluation, e |  no  |
| [Questionnaire](Questionnaire.md) | A questionnaire groups questions |  no  |
| [Term](Term.md) | A term and its definitions |  no  |
| [Action](Action.md) | Action to remediate a risk |  no  |
| [Question](Question.md) | An evaluation where a question has to be answered |  no  |
| [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | Benchmark metadata cards offer a standardized way to document LLM benchmarks ... |  no  |







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
- Action
- AiEval
- BenchmarkMetadataCard
- LLMIntrinsic
range: Risk
multivalued: true
inlined: false
any_of:
- range: RiskConcept
- range: Term

```
</details>
