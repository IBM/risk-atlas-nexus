# Enum: AdapterType



URI: [AdapterType](AdapterType.md)

## Permissible Values

| Value | Meaning | Description |
| --- | --- | --- |
| LORA | None | Low-rank adapters, or LoRAs, are a fast way to give generalist large language... |
| ALORA | None | Activated LoRA (aLoRA) is a low rank adapter architecture that allows for reu... |
| X-LORA | None | Mixture of LoRA Experts (X-LoRA) is a mixture of experts method for LoRA whic... |




## Slots

| Name | Description |
| ---  | --- |
| [hasAdapterType](hasAdapterType.md) | The Adapter type, for example: LORA, ALORA, X-LORA |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology






## LinkML Source

<details>
```yaml
name: AdapterType
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
rank: 1000
permissible_values:
  LORA:
    text: LORA
    description: Low-rank adapters, or LoRAs, are a fast way to give generalist large
      language models targeted knowledge and skills so they can do things like summarize
      IT manuals or rate the accuracy of their own answers. LoRA reduces the number
      of trainable parameters by learning pairs of rank-decompostion matrices while
      freezing the original weights. This vastly reduces the storage requirement for
      large language models adapted to specific tasks and enables efficient task-switching
      during deployment all without introducing inference latency. LoRA also outperforms
      several other adaptation methods including adapter, prefix-tuning, and fine-tuning.
      See arXiv:2106.09685
  ALORA:
    text: ALORA
    description: Activated LoRA (aLoRA) is a low rank adapter architecture that allows
      for reusing existing base model KV cache for more efficient inference, unlike
      standard LoRA models. As a result, aLoRA models can be quickly invoked as-needed
      for specialized tasks during (long) flows where the base model is primarily
      used, avoiding potentially expensive prefill costs in terms of latency, throughput,
      and GPU memory. See arXiv:2504.12397 for further details.
  X-LORA:
    text: X-LORA
    description: Mixture of LoRA Experts (X-LoRA) is a mixture of experts method for
      LoRA which works by using dense or sparse gating to dynamically activate LoRA
      experts.

```
</details>
