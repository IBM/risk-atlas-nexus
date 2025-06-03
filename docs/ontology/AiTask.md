

# Class: AiTask


_A task, such as summarization and classification, performed by an AI._





URI: [airo:AiCapability](https://w3id.org/airo#AiCapability)






```mermaid
 classDiagram
    class AiTask
    click AiTask href "../AiTask"
      Entity <|-- AiTask
        click Entity href "../Entity"

      AiTask : dateCreated

      AiTask : dateModified

      AiTask : description

      AiTask : id

      AiTask : name

      AiTask : url


```





## Inheritance
* [Entity](Entity.md)
    * **AiTask**



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
| [Container](Container.md) | [aitasks](aitasks.md) | range | [AiTask](AiTask.md) |
| [BaseAi](BaseAi.md) | [performsTask](performsTask.md) | range | [AiTask](AiTask.md) |
| [AiSystem](AiSystem.md) | [performsTask](performsTask.md) | range | [AiTask](AiTask.md) |
| [AiAgent](AiAgent.md) | [performsTask](performsTask.md) | range | [AiTask](AiTask.md) |
| [AiModel](AiModel.md) | [performsTask](performsTask.md) | range | [AiTask](AiTask.md) |
| [LargeLanguageModel](LargeLanguageModel.md) | [performsTask](performsTask.md) | range | [AiTask](AiTask.md) |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | airo:AiCapability |
| native | nexus:AiTask |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: AiTask
description: A task, such as summarization and classification, performed by an AI.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
class_uri: airo:AiCapability

```
</details>

### Induced

<details>
```yaml
name: AiTask
description: A task, such as summarization and classification, performed by an AI.
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
    owner: AiTask
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
    owner: AiTask
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
    owner: AiTask
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
    owner: AiTask
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
    owner: AiTask
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
    owner: AiTask
    domain_of:
    - Entity
    range: date
    required: false
class_uri: airo:AiCapability

```
</details>
