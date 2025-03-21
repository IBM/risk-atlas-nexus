

# Class: AiProvider


_A provider under the AI Act is defined by Article 3(3) as a natural or legal person or body that develops an AI system or general-purpose AI model or has an AI system or general-purpose AI model developed; and places that system or model on the market, or puts that system into service, under the provider's own name or trademark, whether for payment or free for charge._





URI: [airo:AIProvider](https://w3id.org/airo#AIProvider)






```mermaid
 classDiagram
    class AiProvider
    click AiProvider href "../AiProvider"
      Organization <|-- AiProvider
        click Organization href "../Organization"
      
      AiProvider : dateCreated
        
      AiProvider : dateModified
        
      AiProvider : description
        
      AiProvider : grants_license
        
          
    
    
    AiProvider --> "0..1" License : grants_license
    click License href "../License"

        
      AiProvider : id
        
      AiProvider : name
        
      AiProvider : url
        
      
```





## Inheritance
* [Entity](Entity.md)
    * [Organization](Organization.md)
        * **AiProvider**



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [grants_license](grants_license.md) | 0..1 <br/> [License](License.md) | A relationship from a granting entity such as an Organization to a License in... | [Organization](Organization.md) |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [BaseAi](BaseAi.md) | [isProvidedBy](isProvidedBy.md) | range | [AiProvider](AiProvider.md) |
| [AiSystem](AiSystem.md) | [isProvidedBy](isProvidedBy.md) | range | [AiProvider](AiProvider.md) |
| [AiAgent](AiAgent.md) | [isProvidedBy](isProvidedBy.md) | range | [AiProvider](AiProvider.md) |
| [AiModel](AiModel.md) | [isProvidedBy](isProvidedBy.md) | range | [AiProvider](AiProvider.md) |
| [LargeLanguageModel](LargeLanguageModel.md) | [isProvidedBy](isProvidedBy.md) | range | [AiProvider](AiProvider.md) |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | airo:AIProvider |
| native | nexus:AiProvider |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: AiProvider
description: A provider under the AI Act is defined by Article 3(3) as a natural or
  legal person or body that develops an AI system or general-purpose AI model or has
  an AI system or general-purpose AI model developed; and places that system or model
  on the market, or puts that system into service, under the provider's own name or
  trademark, whether for payment or free for charge.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Organization
class_uri: airo:AIProvider

```
</details>

### Induced

<details>
```yaml
name: AiProvider
description: A provider under the AI Act is defined by Article 3(3) as a natural or
  legal person or body that develops an AI system or general-purpose AI model or has
  an AI system or general-purpose AI model developed; and places that system or model
  on the market, or puts that system into service, under the provider's own name or
  trademark, whether for payment or free for charge.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Organization
attributes:
  grants_license:
    name: grants_license
    description: A relationship from a granting entity such as an Organization to
      a License instance.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: grants_license
    owner: AiProvider
    domain_of:
    - Organization
    range: License
  id:
    name: id
    description: A unique identifier to this instance of the model element. Example
      identifiers include UUID, URI, URN, etc.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:identifier
    identifier: true
    alias: id
    owner: AiProvider
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
    owner: AiProvider
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
    owner: AiProvider
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
    owner: AiProvider
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
    owner: AiProvider
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
    owner: AiProvider
    domain_of:
    - Entity
    range: date
    required: false
class_uri: airo:AIProvider

```
</details>