# Enum: AiSystemType



URI: [AiSystemType](AiSystemType.md)

## Permissible Values

| Value | Meaning | Description |
| --- | --- | --- |
| GPAI | None | General-purpose AI (GPAI) |
| GPAI_OS | None | General-purpose AI (GPAI) models released under free and open-source licences |
| PROHIBITED | None | Prohibited AI system due to unacceptable risk category (e |
| SCIENTIFIC_RD | None | AI used for scientific research and development |
| MILITARY_SECURITY | None | AI used for military, defense and security purposes |
| HIGH_RISK | None | AI systems pursuant to Article 6(1)(2) Classification Rules for High-Risk AI ... |




## Slots

| Name | Description |
| ---  | --- |
| [hasEuAiSystemType](hasEuAiSystemType.md) | The type of system as defined by the EU AI Act |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology






## LinkML Source

<details>
```yaml
name: AiSystemType
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
permissible_values:
  GPAI:
    text: GPAI
    description: General-purpose AI (GPAI)
  GPAI_OS:
    text: GPAI_OS
    description: General-purpose AI (GPAI) models released under free and open-source
      licences
  PROHIBITED:
    text: PROHIBITED
    description: Prohibited AI system due to unacceptable risk category (e.g. social
      scoring systems and manipulative AI).
  SCIENTIFIC_RD:
    text: SCIENTIFIC_RD
    description: AI used for scientific research and development
  MILITARY_SECURITY:
    text: MILITARY_SECURITY
    description: AI used for military, defense and security purposes.
  HIGH_RISK:
    text: HIGH_RISK
    description: AI systems pursuant to Article 6(1)(2) Classification Rules for High-Risk
      AI Systems

```
</details>
