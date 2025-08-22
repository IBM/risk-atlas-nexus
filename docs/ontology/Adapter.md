

# Class: Adapter


_Adapter-based methods add extra trainable parameters after the attention and fully-connected layers of a frozen pretrained model to reduce memory-usage and speed up training. The adapters are typically small but demonstrate comparable performance to a fully finetuned model and enable training larger models with fewer resources. (https://huggingface.co/docs/peft/en/conceptual_guides/adapter)_





URI: [nexus:Adapter](https://ibm.github.io/risk-atlas-nexus/ontology/Adapter)






```mermaid
 classDiagram
    class Adapter
    click Adapter href "../Adapter"
      Entity <|-- Adapter
        click Entity href "../Entity"

      Adapter : dateCreated

      Adapter : dateModified

      Adapter : description

      Adapter : id

      Adapter : name

      Adapter : url


```





## Inheritance
* [Entity](Entity.md)
    * **Adapter**



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [Container](Container.md) | [adapters](adapters.md) | range | [Adapter](Adapter.md) |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:Adapter |
| native | nexus:Adapter |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: Adapter
description: Adapter-based methods add extra trainable parameters after the attention
  and fully-connected layers of a frozen pretrained model to reduce memory-usage and
  speed up training. The adapters are typically small but demonstrate comparable performance
  to a fully finetuned model and enable training larger models with fewer resources.
  (https://huggingface.co/docs/peft/en/conceptual_guides/adapter)
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity

```
</details>

### Induced

<details>
```yaml
name: Adapter
description: Adapter-based methods add extra trainable parameters after the attention
  and fully-connected layers of a frozen pretrained model to reduce memory-usage and
  speed up training. The adapters are typically small but demonstrate comparable performance
  to a fully finetuned model and enable training larger models with fewer resources.
  (https://huggingface.co/docs/peft/en/conceptual_guides/adapter)
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
attributes:
  id:
    name: id
    description: A unique identifier to this instance of the model element. Example
      identifiers include UUID, URI, URN, etc.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:identifier
    identifier: true
    alias: id
    owner: Adapter
    domain_of:
    - Entity
    range: string
    required: true
  name:
    name: name
    description: A text name of this instance.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:name
    alias: name
    owner: Adapter
    domain_of:
    - Entity
    - BenchmarkMetadataCard
    range: string
  description:
    name: description
    description: The description of an entity
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:description
    alias: description
    owner: Adapter
    domain_of:
    - Entity
    range: string
  url:
    name: url
    description: An optional URL associated with this instance.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:url
    alias: url
    owner: Adapter
    domain_of:
    - Entity
    range: uri
  dateCreated:
    name: dateCreated
    description: The date on which the entity was created.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:dateCreated
    alias: dateCreated
    owner: Adapter
    domain_of:
    - Entity
    range: date
    required: false
  dateModified:
    name: dateModified
    description: The date on which the entity was most recently modified.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:dateModified
    alias: dateModified
    owner: Adapter
    domain_of:
    - Entity
    range: date
    required: false

```
</details>
