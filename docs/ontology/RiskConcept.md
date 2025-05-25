

# Class: RiskConcept


_An umbrella term for refering to risk, risk source, consequence and impact._





URI: [airo:RiskConcept](https://w3id.org/airo#RiskConcept)






```mermaid
 classDiagram
    class RiskConcept
    click RiskConcept href "../RiskConcept"
      Entity <|-- RiskConcept
        click Entity href "../Entity"
      

      RiskConcept <|-- RiskGroup
        click RiskGroup href "../RiskGroup"
      RiskConcept <|-- Risk
        click Risk href "../Risk"
      RiskConcept <|-- RiskControl
        click RiskControl href "../RiskControl"
      RiskConcept <|-- RiskIncident
        click RiskIncident href "../RiskIncident"
      RiskConcept <|-- Impact
        click Impact href "../Impact"
      
      
      RiskConcept : dateCreated
        
      RiskConcept : dateModified
        
      RiskConcept : description
        
      RiskConcept : id
        
      RiskConcept : isDetectedBy
        
          
    
    
    RiskConcept --> "*" RiskControl : isDetectedBy
    click RiskControl href "../RiskControl"

        
      RiskConcept : name
        
      RiskConcept : url
        
      
```





## Inheritance
* [Entity](Entity.md)
    * **RiskConcept**



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [isDetectedBy](isDetectedBy.md) | * <br/> [RiskControl](RiskControl.md) | A relationship where a risk, risk source, consequence, or impact is detected ... | direct |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [RiskGroup](RiskGroup.md) | [isDetectedBy](isDetectedBy.md) | domain | [RiskConcept](RiskConcept.md) |
| [Risk](Risk.md) | [detectsRiskConcept](detectsRiskConcept.md) | range | [RiskConcept](RiskConcept.md) |
| [Risk](Risk.md) | [isDetectedBy](isDetectedBy.md) | domain | [RiskConcept](RiskConcept.md) |
| [RiskConcept](RiskConcept.md) | [isDetectedBy](isDetectedBy.md) | domain | [RiskConcept](RiskConcept.md) |
| [RiskControl](RiskControl.md) | [detectsRiskConcept](detectsRiskConcept.md) | range | [RiskConcept](RiskConcept.md) |
| [RiskControl](RiskControl.md) | [isDetectedBy](isDetectedBy.md) | domain | [RiskConcept](RiskConcept.md) |
| [Action](Action.md) | [hasRelatedRisk](hasRelatedRisk.md) | domain | [RiskConcept](RiskConcept.md) |
| [RiskIncident](RiskIncident.md) | [hasStatus](hasStatus.md) | domain | [RiskConcept](RiskConcept.md) |
| [RiskIncident](RiskIncident.md) | [hasSeverity](hasSeverity.md) | domain | [RiskConcept](RiskConcept.md) |
| [RiskIncident](RiskIncident.md) | [hasLikelihood](hasLikelihood.md) | domain | [RiskConcept](RiskConcept.md) |
| [RiskIncident](RiskIncident.md) | [hasImpactOn](hasImpactOn.md) | domain | [RiskConcept](RiskConcept.md) |
| [RiskIncident](RiskIncident.md) | [hasConsequence](hasConsequence.md) | domain | [RiskConcept](RiskConcept.md) |
| [RiskIncident](RiskIncident.md) | [hasImpact](hasImpact.md) | domain | [RiskConcept](RiskConcept.md) |
| [RiskIncident](RiskIncident.md) | [isDetectedBy](isDetectedBy.md) | domain | [RiskConcept](RiskConcept.md) |
| [Impact](Impact.md) | [isDetectedBy](isDetectedBy.md) | domain | [RiskConcept](RiskConcept.md) |
| [AiEval](AiEval.md) | [hasRelatedRisk](hasRelatedRisk.md) | domain | [RiskConcept](RiskConcept.md) |
| [Question](Question.md) | [hasRelatedRisk](hasRelatedRisk.md) | domain | [RiskConcept](RiskConcept.md) |
| [Questionnaire](Questionnaire.md) | [hasRelatedRisk](hasRelatedRisk.md) | domain | [RiskConcept](RiskConcept.md) |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | airo:RiskConcept |
| native | nexus:RiskConcept |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: RiskConcept
description: An umbrella term for refering to risk, risk source, consequence and impact.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
slots:
- isDetectedBy
class_uri: airo:RiskConcept

```
</details>

### Induced

<details>
```yaml
name: RiskConcept
description: An umbrella term for refering to risk, risk source, consequence and impact.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Entity
attributes:
  isDetectedBy:
    name: isDetectedBy
    description: A relationship where a risk, risk source, consequence, or impact
      is detected by a risk control.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain: RiskConcept
    alias: isDetectedBy
    owner: RiskConcept
    domain_of:
    - RiskConcept
    inverse: detectsRiskConcept
    range: RiskControl
    multivalued: true
    inlined: false
  id:
    name: id
    description: A unique identifier to this instance of the model element. Example
      identifiers include UUID, URI, URN, etc.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:identifier
    identifier: true
    alias: id
    owner: RiskConcept
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
    owner: RiskConcept
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
    owner: RiskConcept
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
    owner: RiskConcept
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
    owner: RiskConcept
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
    owner: RiskConcept
    domain_of:
    - Entity
    range: date
    required: false
class_uri: airo:RiskConcept

```
</details>