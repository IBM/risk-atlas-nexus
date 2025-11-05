

# Class: Container


_An umbrella object that holds the ontology class instances_





URI: [nexus:Container](https://ibm.github.io/risk-atlas-nexus/ontology/Container)






```mermaid
 classDiagram
    class Container
    click Container href "../Container"
      Container : actions





        Container --> "*" Action : actions
        click Action href "../Action"



      Container : adapters





        Container --> "*" Adapter : adapters
        click Adapter href "../Adapter"



      Container : aievalresults





        Container --> "*" AiEvalResult : aievalresults
        click AiEvalResult href "../AiEvalResult"



      Container : aimodelfamilies





        Container --> "*" LargeLanguageModelFamily : aimodelfamilies
        click LargeLanguageModelFamily href "../LargeLanguageModelFamily"



      Container : aimodels





        Container --> "*" LargeLanguageModel : aimodels
        click LargeLanguageModel href "../LargeLanguageModel"



      Container : aitasks





        Container --> "*" AiTask : aitasks
        click AiTask href "../AiTask"



      Container : benchmarkmetadatacards





        Container --> "*" BenchmarkMetadataCard : benchmarkmetadatacards
        click BenchmarkMetadataCard href "../BenchmarkMetadataCard"



      Container : datasets





        Container --> "*" Dataset : datasets
        click Dataset href "../Dataset"



      Container : documents





        Container --> "*" Documentation : documents
        click Documentation href "../Documentation"



      Container : evaluations





        Container --> "*" AiEval : evaluations
        click AiEval href "../AiEval"



      Container : licenses





        Container --> "*" License : licenses
        click License href "../License"



      Container : llmintrinsics





        Container --> "*" LLMIntrinsic : llmintrinsics
        click LLMIntrinsic href "../LLMIntrinsic"



      Container : llmquestionpolicies





        Container --> "*" LLMQuestionPolicy : llmquestionpolicies
        click LLMQuestionPolicy href "../LLMQuestionPolicy"



      Container : modalities





        Container --> "*" Modality : modalities
        click Modality href "../Modality"



      Container : obligations





        Container --> "*" Obligation : obligations
        click Obligation href "../Obligation"



      Container : organizations





        Container --> "*" Organization : organizations
        click Organization href "../Organization"



      Container : permissions





        Container --> "*" Permission : permissions
        click Permission href "../Permission"



      Container : policies





        Container --> "*" Policy : policies
        click Policy href "../Policy"



      Container : principles





        Container --> "*" Principle : principles
        click Principle href "../Principle"



      Container : prohibitions





        Container --> "*" Prohibition : prohibitions
        click Prohibition href "../Prohibition"



      Container : riskcontrols





        Container --> "*" RiskControl : riskcontrols
        click RiskControl href "../RiskControl"



      Container : riskgroups





        Container --> "*" RiskGroup : riskgroups
        click RiskGroup href "../RiskGroup"



      Container : riskincidents





        Container --> "*" RiskIncident : riskincidents
        click RiskIncident href "../RiskIncident"



      Container : risks





        Container --> "*" Risk : risks
        click Risk href "../Risk"



      Container : rules





        Container --> "*" Rule : rules
        click Rule href "../Rule"



      Container : stakeholdergroups





        Container --> "*" StakeholderGroup : stakeholdergroups
        click StakeholderGroup href "../StakeholderGroup"



      Container : stakeholders





        Container --> "*" Stakeholder : stakeholders
        click Stakeholder href "../Stakeholder"



      Container : taxonomies





        Container --> "*" RiskTaxonomy : taxonomies
        click RiskTaxonomy href "../RiskTaxonomy"



      Container : terms





        Container --> "*" Term : terms
        click Term href "../Term"



      Container : vocabularies





        Container --> "*" Vocabulary : vocabularies
        click Vocabulary href "../Vocabulary"




```




<!-- no inheritance hierarchy -->


## Slots

| Name | Cardinality and Range | Description | Inheritance |
| ---  | --- | --- | --- |
| [organizations](organizations.md) | * <br/> [Organization](Organization.md) | A list of organizations | direct |
| [licenses](licenses.md) | * <br/> [License](License.md) | A list of licenses | direct |
| [modalities](modalities.md) | * <br/> [Modality](Modality.md) | A list of AI modalities | direct |
| [aitasks](aitasks.md) | * <br/> [AiTask](AiTask.md) | A list of AI tasks | direct |
| [documents](documents.md) | * <br/> [Documentation](Documentation.md) | A list of documents | direct |
| [datasets](datasets.md) | * <br/> [Dataset](Dataset.md) | A list of data sets | direct |
| [llmintrinsics](llmintrinsics.md) | * <br/> [LLMIntrinsic](LLMIntrinsic.md) | A list of LLMIntrinsics | direct |
| [adapters](adapters.md) | * <br/> [Adapter](Adapter.md) | A list of Adapters | direct |
| [taxonomies](taxonomies.md) | * <br/> [RiskTaxonomy](RiskTaxonomy.md) | A list of AI risk taxonomies | direct |
| [vocabularies](vocabularies.md) | * <br/> [Vocabulary](Vocabulary.md) | A list of vocabularies | direct |
| [riskgroups](riskgroups.md) | * <br/> [RiskGroup](RiskGroup.md) | A list of AI risk groups | direct |
| [risks](risks.md) | * <br/> [Risk](Risk.md) | A list of AI risks | direct |
| [riskcontrols](riskcontrols.md) | * <br/> [RiskControl](RiskControl.md) | A list of AI risk controls | direct |
| [riskincidents](riskincidents.md) | * <br/> [RiskIncident](RiskIncident.md) | A list of AI risk incidents | direct |
| [terms](terms.md) | * <br/> [Term](Term.md) | A list of terms from a vocabulary | direct |
| [stakeholdergroups](stakeholdergroups.md) | * <br/> [StakeholderGroup](StakeholderGroup.md) | A list of AI stakeholder groups | direct |
| [stakeholders](stakeholders.md) | * <br/> [Stakeholder](Stakeholder.md) | A list of stakeholders | direct |
| [actions](actions.md) | * <br/> [Action](Action.md) | A list of risk related actions | direct |
| [evaluations](evaluations.md) | * <br/> [AiEval](AiEval.md) | A list of AI evaluation methods | direct |
| [aievalresults](aievalresults.md) | * <br/> [AiEvalResult](AiEvalResult.md) | A list of AI evaluation results | direct |
| [benchmarkmetadatacards](benchmarkmetadatacards.md) | * <br/> [BenchmarkMetadataCard](BenchmarkMetadataCard.md) | A list of AI evaluation benchmark metadata cards | direct |
| [aimodelfamilies](aimodelfamilies.md) | * <br/> [LargeLanguageModelFamily](LargeLanguageModelFamily.md) | A list of AI model families | direct |
| [aimodels](aimodels.md) | * <br/> [LargeLanguageModel](LargeLanguageModel.md) | A list of AI models | direct |
| [policies](policies.md) | * <br/> [Policy](Policy.md) | A list of policies | direct |
| [llmquestionpolicies](llmquestionpolicies.md) | * <br/> [LLMQuestionPolicy](LLMQuestionPolicy.md) | A list of LLM question policies | direct |
| [rules](rules.md) | * <br/> [Rule](Rule.md) | A list of rules | direct |
| [prohibitions](prohibitions.md) | * <br/> [Prohibition](Prohibition.md) | A list of prohibitions | direct |
| [permissions](permissions.md) | * <br/> [Permission](Permission.md) | A list of Permissions | direct |
| [obligations](obligations.md) | * <br/> [Obligation](Obligation.md) | A list of Obligations | direct |
| [principles](principles.md) | * <br/> [Principle](Principle.md) | A list of Principles | direct |









## Identifier and Mapping Information







### Schema Source


* from schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology




## Mappings

| Mapping Type | Mapped Value |
| ---  | ---  |
| self | nexus:Container |
| native | nexus:Container |







## LinkML Source

<!-- TODO: investigate https://stackoverflow.com/questions/37606292/how-to-create-tabbed-code-blocks-in-mkdocs-or-sphinx -->

### Direct

<details>
```yaml
name: Container
description: An umbrella object that holds the ontology class instances
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
attributes:
  organizations:
    name: organizations
    description: A list of organizations
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Organization
    multivalued: true
    inlined: true
    inlined_as_list: true
  licenses:
    name: licenses
    description: A list of licenses
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: License
    multivalued: true
    inlined: true
    inlined_as_list: true
  modalities:
    name: modalities
    description: A list of AI modalities
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Modality
    multivalued: true
    inlined: true
    inlined_as_list: true
  aitasks:
    name: aitasks
    description: A list of AI tasks
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: AiTask
    multivalued: true
    inlined: true
    inlined_as_list: true
  documents:
    name: documents
    description: A list of documents
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Documentation
    multivalued: true
    inlined: true
    inlined_as_list: true
  datasets:
    name: datasets
    description: A list of data sets
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Dataset
    multivalued: true
    inlined: true
    inlined_as_list: true
  llmintrinsics:
    name: llmintrinsics
    description: A list of LLMIntrinsics
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: LLMIntrinsic
    multivalued: true
    inlined: true
    inlined_as_list: true
  adapters:
    name: adapters
    description: A list of Adapters
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Adapter
    multivalued: true
    inlined: true
    inlined_as_list: true
  taxonomies:
    name: taxonomies
    description: A list of AI risk taxonomies
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: RiskTaxonomy
    multivalued: true
    inlined: true
    inlined_as_list: true
  vocabularies:
    name: vocabularies
    description: A list of vocabularies
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Vocabulary
    multivalued: true
    inlined: true
    inlined_as_list: true
  riskgroups:
    name: riskgroups
    description: A list of AI risk groups
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: RiskGroup
    multivalued: true
    inlined: true
    inlined_as_list: true
  risks:
    name: risks
    description: A list of AI risks
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Risk
    multivalued: true
    inlined: true
    inlined_as_list: true
  riskcontrols:
    name: riskcontrols
    description: A list of AI risk controls
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: RiskControl
    multivalued: true
    inlined: true
    inlined_as_list: true
  riskincidents:
    name: riskincidents
    description: A list of AI risk incidents
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: RiskIncident
    multivalued: true
    inlined: true
    inlined_as_list: true
  terms:
    name: terms
    description: A list of terms from a vocabulary
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Term
    multivalued: true
    inlined: true
    inlined_as_list: true
  stakeholdergroups:
    name: stakeholdergroups
    description: A list of AI stakeholder groups
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: StakeholderGroup
    multivalued: true
    inlined: true
    inlined_as_list: true
  stakeholders:
    name: stakeholders
    description: A list of stakeholders
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Stakeholder
    multivalued: true
    inlined: true
    inlined_as_list: true
  actions:
    name: actions
    description: A list of risk related actions
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Action
    multivalued: true
    inlined: true
    inlined_as_list: true
  evaluations:
    name: evaluations
    description: A list of AI evaluation methods
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: AiEval
    multivalued: true
    inlined: true
    inlined_as_list: true
  aievalresults:
    name: aievalresults
    description: A list of AI evaluation results
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: AiEvalResult
    multivalued: true
    inlined: true
    inlined_as_list: true
  benchmarkmetadatacards:
    name: benchmarkmetadatacards
    description: A list of AI evaluation benchmark metadata cards
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: BenchmarkMetadataCard
    multivalued: true
    inlined: true
    inlined_as_list: true
  aimodelfamilies:
    name: aimodelfamilies
    description: A list of AI model families
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: LargeLanguageModelFamily
    multivalued: true
    inlined: true
    inlined_as_list: true
  aimodels:
    name: aimodels
    description: A list of AI models
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: LargeLanguageModel
    multivalued: true
    inlined: true
    inlined_as_list: true
  policies:
    name: policies
    description: A list of policies
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Policy
    multivalued: true
    inlined: true
    inlined_as_list: true
  llmquestionpolicies:
    name: llmquestionpolicies
    description: A list of LLM question policies
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: LLMQuestionPolicy
    multivalued: true
    inlined: true
    inlined_as_list: true
  rules:
    name: rules
    description: A list of rules
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Rule
    multivalued: true
    inlined: true
    inlined_as_list: true
  prohibitions:
    name: prohibitions
    description: A list of prohibitions
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Prohibition
    multivalued: true
    inlined: true
    inlined_as_list: true
  permissions:
    name: permissions
    description: A list of Permissions
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Permission
    multivalued: true
    inlined: true
    inlined_as_list: true
  obligations:
    name: obligations
    description: A list of Obligations
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Obligation
    multivalued: true
    inlined: true
    inlined_as_list: true
  principles:
    name: principles
    description: A list of Principles
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    domain_of:
    - Container
    range: Principle
    multivalued: true
    inlined: true
    inlined_as_list: true
tree_root: true

```
</details>

### Induced

<details>
```yaml
name: Container
description: An umbrella object that holds the ontology class instances
from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
attributes:
  organizations:
    name: organizations
    description: A list of organizations
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: organizations
    owner: Container
    domain_of:
    - Container
    range: Organization
    multivalued: true
    inlined: true
    inlined_as_list: true
  licenses:
    name: licenses
    description: A list of licenses
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: licenses
    owner: Container
    domain_of:
    - Container
    range: License
    multivalued: true
    inlined: true
    inlined_as_list: true
  modalities:
    name: modalities
    description: A list of AI modalities
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: modalities
    owner: Container
    domain_of:
    - Container
    range: Modality
    multivalued: true
    inlined: true
    inlined_as_list: true
  aitasks:
    name: aitasks
    description: A list of AI tasks
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: aitasks
    owner: Container
    domain_of:
    - Container
    range: AiTask
    multivalued: true
    inlined: true
    inlined_as_list: true
  documents:
    name: documents
    description: A list of documents
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: documents
    owner: Container
    domain_of:
    - Container
    range: Documentation
    multivalued: true
    inlined: true
    inlined_as_list: true
  datasets:
    name: datasets
    description: A list of data sets
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: datasets
    owner: Container
    domain_of:
    - Container
    range: Dataset
    multivalued: true
    inlined: true
    inlined_as_list: true
  llmintrinsics:
    name: llmintrinsics
    description: A list of LLMIntrinsics
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: llmintrinsics
    owner: Container
    domain_of:
    - Container
    range: LLMIntrinsic
    multivalued: true
    inlined: true
    inlined_as_list: true
  adapters:
    name: adapters
    description: A list of Adapters
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: adapters
    owner: Container
    domain_of:
    - Container
    range: Adapter
    multivalued: true
    inlined: true
    inlined_as_list: true
  taxonomies:
    name: taxonomies
    description: A list of AI risk taxonomies
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: taxonomies
    owner: Container
    domain_of:
    - Container
    range: RiskTaxonomy
    multivalued: true
    inlined: true
    inlined_as_list: true
  vocabularies:
    name: vocabularies
    description: A list of vocabularies
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: vocabularies
    owner: Container
    domain_of:
    - Container
    range: Vocabulary
    multivalued: true
    inlined: true
    inlined_as_list: true
  riskgroups:
    name: riskgroups
    description: A list of AI risk groups
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: riskgroups
    owner: Container
    domain_of:
    - Container
    range: RiskGroup
    multivalued: true
    inlined: true
    inlined_as_list: true
  risks:
    name: risks
    description: A list of AI risks
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: risks
    owner: Container
    domain_of:
    - Container
    range: Risk
    multivalued: true
    inlined: true
    inlined_as_list: true
  riskcontrols:
    name: riskcontrols
    description: A list of AI risk controls
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: riskcontrols
    owner: Container
    domain_of:
    - Container
    range: RiskControl
    multivalued: true
    inlined: true
    inlined_as_list: true
  riskincidents:
    name: riskincidents
    description: A list of AI risk incidents
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: riskincidents
    owner: Container
    domain_of:
    - Container
    range: RiskIncident
    multivalued: true
    inlined: true
    inlined_as_list: true
  terms:
    name: terms
    description: A list of terms from a vocabulary
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: terms
    owner: Container
    domain_of:
    - Container
    range: Term
    multivalued: true
    inlined: true
    inlined_as_list: true
  stakeholdergroups:
    name: stakeholdergroups
    description: A list of AI stakeholder groups
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: stakeholdergroups
    owner: Container
    domain_of:
    - Container
    range: StakeholderGroup
    multivalued: true
    inlined: true
    inlined_as_list: true
  stakeholders:
    name: stakeholders
    description: A list of stakeholders
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: stakeholders
    owner: Container
    domain_of:
    - Container
    range: Stakeholder
    multivalued: true
    inlined: true
    inlined_as_list: true
  actions:
    name: actions
    description: A list of risk related actions
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: actions
    owner: Container
    domain_of:
    - Container
    range: Action
    multivalued: true
    inlined: true
    inlined_as_list: true
  evaluations:
    name: evaluations
    description: A list of AI evaluation methods
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: evaluations
    owner: Container
    domain_of:
    - Container
    range: AiEval
    multivalued: true
    inlined: true
    inlined_as_list: true
  aievalresults:
    name: aievalresults
    description: A list of AI evaluation results
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: aievalresults
    owner: Container
    domain_of:
    - Container
    range: AiEvalResult
    multivalued: true
    inlined: true
    inlined_as_list: true
  benchmarkmetadatacards:
    name: benchmarkmetadatacards
    description: A list of AI evaluation benchmark metadata cards
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: benchmarkmetadatacards
    owner: Container
    domain_of:
    - Container
    range: BenchmarkMetadataCard
    multivalued: true
    inlined: true
    inlined_as_list: true
  aimodelfamilies:
    name: aimodelfamilies
    description: A list of AI model families
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: aimodelfamilies
    owner: Container
    domain_of:
    - Container
    range: LargeLanguageModelFamily
    multivalued: true
    inlined: true
    inlined_as_list: true
  aimodels:
    name: aimodels
    description: A list of AI models
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: aimodels
    owner: Container
    domain_of:
    - Container
    range: LargeLanguageModel
    multivalued: true
    inlined: true
    inlined_as_list: true
  policies:
    name: policies
    description: A list of policies
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: policies
    owner: Container
    domain_of:
    - Container
    range: Policy
    multivalued: true
    inlined: true
    inlined_as_list: true
  llmquestionpolicies:
    name: llmquestionpolicies
    description: A list of LLM question policies
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: llmquestionpolicies
    owner: Container
    domain_of:
    - Container
    range: LLMQuestionPolicy
    multivalued: true
    inlined: true
    inlined_as_list: true
  rules:
    name: rules
    description: A list of rules
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: rules
    owner: Container
    domain_of:
    - Container
    range: Rule
    multivalued: true
    inlined: true
    inlined_as_list: true
  prohibitions:
    name: prohibitions
    description: A list of prohibitions
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: prohibitions
    owner: Container
    domain_of:
    - Container
    range: Prohibition
    multivalued: true
    inlined: true
    inlined_as_list: true
  permissions:
    name: permissions
    description: A list of Permissions
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: permissions
    owner: Container
    domain_of:
    - Container
    range: Permission
    multivalued: true
    inlined: true
    inlined_as_list: true
  obligations:
    name: obligations
    description: A list of Obligations
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: obligations
    owner: Container
    domain_of:
    - Container
    range: Obligation
    multivalued: true
    inlined: true
    inlined_as_list: true
  principles:
    name: principles
    description: A list of Principles
    from_schema: https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology
    rank: 1000
    alias: principles
    owner: Container
    domain_of:
    - Container
    range: Principle
    multivalued: true
    inlined: true
    inlined_as_list: true
tree_root: true

```
</details>
