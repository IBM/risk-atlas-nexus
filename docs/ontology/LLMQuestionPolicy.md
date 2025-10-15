

# Class: LLMQuestionPolicy


_The policy guides how the language model should answer a diverse set of sensitive questions._





URI: [nexus:LLMQuestionPolicy](https://ibm.github.io/risk-atlas-nexus/ontology/LLMQuestionPolicy)






```mermaid
 classDiagram
    class LLMQuestionPolicy
    click LLMQuestionPolicy href "../LLMQuestionPolicy"
      Policy <|-- LLMQuestionPolicy
        click Policy href "../Policy"

      LLMQuestionPolicy : dateCreated

      LLMQuestionPolicy : dateModified

      LLMQuestionPolicy : description

      LLMQuestionPolicy : hasException

      LLMQuestionPolicy : hasReasonDenial

      LLMQuestionPolicy : hasRelatedRisk





        LLMQuestionPolicy --> "*" Risk : hasRelatedRisk
        click Risk href "../Risk"



      LLMQuestionPolicy : hasRule





        LLMQuestionPolicy --> "*" Rule : hasRule
        click Rule href "../Rule"



      LLMQuestionPolicy : hasShortReplyType

      LLMQuestionPolicy : id

      LLMQuestionPolicy : isDefinedByTaxonomy





        LLMQuestionPolicy --> "0..1" RiskTaxonomy : isDefinedByTaxonomy
        click RiskTaxonomy href "../RiskTaxonomy"



      LLMQuestionPolicy : name

      LLMQuestionPolicy : url


```





## Inheritance
* [Entity](Entity.md)
    * [Policy](Policy.md)
        * **LLMQuestionPolicy**



## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [hasRelatedRisk](hasRelatedRisk.md) | * <br/> [Risk](Risk.md)&nbsp;or&nbsp;<br />[RiskConcept](RiskConcept.md)&nbsp;or&nbsp;<br />[Term](Term.md) | A relationship where an entity relates to a risk | direct |
| [hasRule](hasRule.md) | * <br/> [Rule](Rule.md) | Specifying applicability or inclusion of a rule within specified context | direct |
| [hasReasonDenial](hasReasonDenial.md) | 0..1 <br/> [String](String.md) | Reason for denial | direct |
| [hasShortReplyType](hasShortReplyType.md) | 0..1 <br/> [String](String.md) | Short reply type | direct |
| [hasException](hasException.md) | 0..1 <br/> [String](String.md) | Exception type | direct |
| [isDefinedByTaxonomy](isDefinedByTaxonomy.md) | 0..1 <br/> [RiskTaxonomy](RiskTaxonomy.md) | A relationship where a risk or a risk group is defined by a risk taxonomy | [Policy](Policy.md) |
| [id](id.md) | 1 <br/> [String](String.md) | A unique identifier to this instance of the model element | [Entity](Entity.md) |
| [name](name.md) | 0..1 <br/> [String](String.md) | A text name of this instance | [Entity](Entity.md) |
| [description](description.md) | 0..1 <br/> [String](String.md) | The description of an entity | [Entity](Entity.md) |
| [url](url.md) | 0..1 <br/> [Uri](Uri.md) | An optional URL associated with this instance | [Entity](Entity.md) |
| [dateCreated](dateCreated.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was created | [Entity](Entity.md) |
| [dateModified](dateModified.md) | 0..1 <br/> [Date](Date.md) | The date on which the entity was most recently modified | [Entity](Entity.md) |





## Usages

| used by | used in | type | used |
| ---  | --- | --- | --- |
| [Container](Container.md) | [llmquestionpolicies](llmquestionpolicies.md) | range | [LLMQuestionPolicy](LLMQuestionPolicy.md) |






## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:LLMQuestionPolicy |
| native | nexus:LLMQuestionPolicy |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: LLMQuestionPolicy
description: The policy guides how the language model should answer a diverse set
  of sensitive questions.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Policy
slots:
- hasRelatedRisk
- hasRule
- hasReasonDenial
- hasShortReplyType
- hasException

```
</details>

### Induced

<details>
```yaml
name: LLMQuestionPolicy
description: The policy guides how the language model should answer a diverse set
  of sensitive questions.
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
is_a: Policy
attributes:
  hasRelatedRisk:
    name: hasRelatedRisk
    description: A relationship where an entity relates to a risk
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain: Any
    alias: hasRelatedRisk
    owner: LLMQuestionPolicy
    domain_of:
    - Term
    - LLMQuestionPolicy
    - Action
    - AiEval
    - BenchmarkMetadataCard
    - Adapter
    - LLMIntrinsic
    range: Risk
    multivalued: true
    inlined: false
    any_of:
    - range: RiskConcept
    - range: Term
  hasRule:
    name: hasRule
    description: Specifying applicability or inclusion of a rule within specified
      context.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: dpv:hasRule
    alias: hasRule
    owner: LLMQuestionPolicy
    domain_of:
    - LLMQuestionPolicy
    range: Rule
    multivalued: true
    inlined: false
  hasReasonDenial:
    name: hasReasonDenial
    description: Reason for denial
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: nexus:hasReasonDenial
    alias: hasReasonDenial
    owner: LLMQuestionPolicy
    domain_of:
    - LLMQuestionPolicy
    range: string
    multivalued: false
    inlined: true
  hasShortReplyType:
    name: hasShortReplyType
    description: Short reply type
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: nexus:hasShortReplyType
    alias: hasShortReplyType
    owner: LLMQuestionPolicy
    domain_of:
    - LLMQuestionPolicy
    range: string
    multivalued: false
    inlined: true
  hasException:
    name: hasException
    description: Exception type
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: nexus:hasException
    alias: hasException
    owner: LLMQuestionPolicy
    domain_of:
    - LLMQuestionPolicy
    range: string
    multivalued: false
    inlined: true
  isDefinedByTaxonomy:
    name: isDefinedByTaxonomy
    description: A relationship where a risk or a risk group is defined by a risk
      taxonomy
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:isPartOf
    alias: isDefinedByTaxonomy
    owner: LLMQuestionPolicy
    domain_of:
    - Policy
    - RiskGroup
    - Risk
    - RiskControl
    - Action
    - RiskIncident
    - StakeholderGroup
    - Stakeholder
    range: RiskTaxonomy
  id:
    name: id
    description: A unique identifier to this instance of the model element. Example
      identifiers include UUID, URI, URN, etc.
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    slot_uri: schema:identifier
    identifier: true
    alias: id
    owner: LLMQuestionPolicy
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
    owner: LLMQuestionPolicy
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
    owner: LLMQuestionPolicy
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
    owner: LLMQuestionPolicy
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
    owner: LLMQuestionPolicy
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
    owner: LLMQuestionPolicy
    domain_of:
    - Entity
    range: date
    required: false

```
</details>
