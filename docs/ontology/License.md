

# Class: License


_The general notion of a license which defines terms and grants permissions to users of AI systems, datasets and software._





URI: [airo:License](https://w3id.org/airo#License)






```mermaid
 classDiagram
    class License
    click License href "../License"
      Entity <|-- License
        click Entity href "../Entity"

      License : dateCreated

      License : dateModified

      License : description

      License : id

      License : name

      License : url

      License : version


```





## Inheritance
* [Entity](Entity.md)
    * **License**



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [version](version.md) | 0..1 <br/> [String](String.md) | The version of the entity embodied by a specified resource | direct |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [Container](Container.md) | [licenses](licenses.md) | range | [License](License.md) |
| [Organization](Organization.md) | [grants_license](grants_license.md) | range | [License](License.md) |
| [Dataset](Dataset.md) | [hasLicense](hasLicense.md) | range | [License](License.md) |
| [Documentation](Documentation.md) | [hasLicense](hasLicense.md) | range | [License](License.md) |
| [RiskTaxonomy](RiskTaxonomy.md) | [hasLicense](hasLicense.md) | range | [License](License.md) |
| [AiEval](AiEval.md) | [hasLicense](hasLicense.md) | range | [License](License.md) |
| [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | [hasLicense](hasLicense.md) | range | [License](License.md) |
| [Question](Question.md) | [hasLicense](hasLicense.md) | range | [License](License.md) |
| [Questionnaire](Questionnaire.md) | [hasLicense](hasLicense.md) | range | [License](License.md) |
| [AiOffice](AiOffice.md) | [grants_license](grants_license.md) | range | [License](License.md) |
| [BaseAi](BaseAi.md) | [hasLicense](hasLicense.md) | range | [License](License.md) |
| [AiSystem](AiSystem.md) | [hasLicense](hasLicense.md) | range | [License](License.md) |
| [AiAgent](AiAgent.md) | [hasLicense](hasLicense.md) | range | [License](License.md) |
| [AiModel](AiModel.md) | [hasLicense](hasLicense.md) | range | [License](License.md) |
| [LargeLanguageModel](LargeLanguageModel.md) | [hasLicense](hasLicense.md) | range | [License](License.md) |
| [AiProvider](AiProvider.md) | [grants_license](grants_license.md) | range | [License](License.md) |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | airo:License |
| native | nexus:License |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: License
description: The general notion of a license which defines terms and grants permissions
  to users of AI systems, datasets and software.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
slots:
- version
class_uri: airo:License

```
</details>

### Induced

<details>
```yaml
name: License
description: The general notion of a license which defines terms and grants permissions
  to users of AI systems, datasets and software.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
attributes:
  version:
    name: version
    description: The version of the entity embodied by a specified resource.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:version
    alias: version
    owner: License
    domain_of:
    - License
    - RiskTaxonomy
    range: string
  id:
    name: id
    description: A unique identifier to this instance of the model element. Example
      identifiers include UUID, URI, URN, etc.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:identifier
    identifier: true
    alias: id
    owner: License
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
    owner: License
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
    owner: License
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
    owner: License
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
    owner: License
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
    owner: License
    domain_of:
    - Entity
    range: date
    required: false
class_uri: airo:License

```
</details>
