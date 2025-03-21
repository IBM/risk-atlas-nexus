

# Class: AiLifecyclePhase


_A Phase of AI lifecycle which indicates evolution of the system from conception through retirement._




* __NOTE__: this is an abstract class and should not be instantiated directly


URI: [airo:AILifecyclePhase](https://w3id.org/airo#AILifecyclePhase)






```mermaid
 classDiagram
    class AiLifecyclePhase
    click AiLifecyclePhase href "../AiLifecyclePhase"
      Entity <|-- AiLifecyclePhase
        click Entity href "../Entity"
      

      AiLifecyclePhase <|-- DataPreprocessing
        click DataPreprocessing href "../DataPreprocessing"
      AiLifecyclePhase <|-- AiModelValidation
        click AiModelValidation href "../AiModelValidation"
      
      
      AiLifecyclePhase : dateCreated
        
      AiLifecyclePhase : dateModified
        
      AiLifecyclePhase : description
        
      AiLifecyclePhase : id
        
      AiLifecyclePhase : name
        
      AiLifecyclePhase : url
        
      
```





## Inheritance
* [Entity](Entity.md)
    * **AiLifecyclePhase**
        * [DataPreprocessing](DataPreprocessing.md)
        * [AiModelValidation](AiModelValidation.md)



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |









## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | airo:AILifecyclePhase |
| native | nexus:AiLifecyclePhase |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: AiLifecyclePhase
description: A Phase of AI lifecycle which indicates evolution of the system from
  conception through retirement.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
abstract: true
class_uri: airo:AILifecyclePhase

```
</details>

### Induced

<details>
```yaml
name: AiLifecyclePhase
description: A Phase of AI lifecycle which indicates evolution of the system from
  conception through retirement.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
abstract: true
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
    owner: AiLifecyclePhase
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
    owner: AiLifecyclePhase
    domain_of:
    - Entity
    range: string
  description:
    name: description
    description: The description of an entity
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:description
    alias: description
    owner: AiLifecyclePhase
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
    owner: AiLifecyclePhase
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
    owner: AiLifecyclePhase
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
    owner: AiLifecyclePhase
    domain_of:
    - Entity
    range: date
    required: false
class_uri: airo:AILifecyclePhase

```
</details>