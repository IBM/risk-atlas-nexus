

# Class: Organization


_Any organizational entity such as a corporation, educational institution, consortium, government, etc._





URI: [schema:Organization](http://schema.org/Organization)






```mermaid
 classDiagram
    class Organization
    click Organization href "../Organization"
      Entity <|-- Organization
        click Entity href "../Entity"


      Organization <|-- AiOffice
        click AiOffice href "../AiOffice"
      Organization <|-- AiProvider
        click AiProvider href "../AiProvider"


      Organization : dateCreated

      Organization : dateModified

      Organization : description

      Organization : grants_license




    Organization --> "0..1" License : grants_license
    click License href "../License"


      Organization : id

      Organization : name

      Organization : url


```





## Inheritance
* [Entity](Entity.md)
    * **Organization**
        * [AiOffice](AiOffice.md)
        * [AiProvider](AiProvider.md)



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [grants_license](grants_license.md) | 0..1 <br/> [License](License.md) | A relationship from a granting entity such as an Organization to a License in... | direct |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [Container](Container.md) | [organizations](organizations.md) | range | [Organization](Organization.md) |
| [Dataset](Dataset.md) | [provider](provider.md) | range | [Organization](Organization.md) |
| [BaseAi](BaseAi.md) | [producer](producer.md) | range | [Organization](Organization.md) |
| [AiSystem](AiSystem.md) | [producer](producer.md) | range | [Organization](Organization.md) |
| [AiAgent](AiAgent.md) | [producer](producer.md) | range | [Organization](Organization.md) |
| [AiModel](AiModel.md) | [producer](producer.md) | range | [Organization](Organization.md) |
| [LargeLanguageModel](LargeLanguageModel.md) | [producer](producer.md) | range | [Organization](Organization.md) |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | schema:Organization |
| native | nexus:Organization |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: Organization
description: Any organizational entity such as a corporation, educational institution,
  consortium, government, etc.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
slots:
- grants_license
class_uri: schema:Organization

```
</details>

### Induced

<details>
```yaml
name: Organization
description: Any organizational entity such as a corporation, educational institution,
  consortium, government, etc.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
attributes:
  grants_license:
    name: grants_license
    description: A relationship from a granting entity such as an Organization to
      a License instance.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: grants_license
    owner: Organization
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
    owner: Organization
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
    owner: Organization
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
    owner: Organization
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
    owner: Organization
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
    owner: Organization
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
    owner: Organization
    domain_of:
    - Entity
    range: date
    required: false
class_uri: schema:Organization

```
</details>
