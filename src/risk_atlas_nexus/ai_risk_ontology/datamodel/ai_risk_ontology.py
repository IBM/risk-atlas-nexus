from __future__ import annotations 

import re
import sys
from datetime import (
    date,
    datetime,
    time
)
from decimal import Decimal 
from enum import Enum 
from typing import (
    Any,
    ClassVar,
    Dict,
    List,
    Literal,
    Optional,
    Union
)

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator
)


metamodel_version = "None"
version = "0.5.0"


class ConfiguredBaseModel(BaseModel):
    model_config = ConfigDict(
        validate_assignment = True,
        validate_default = True,
        extra = "forbid",
        arbitrary_types_allowed = True,
        use_enum_values = True,
        strict = False,
    )
    pass




class LinkMLMeta(RootModel):
    root: Dict[str, Any] = {}
    model_config = ConfigDict(frozen=True)

    def __getattr__(self, key:str):
        return getattr(self.root, key)

    def __getitem__(self, key:str):
        return self.root[key]

    def __setitem__(self, key:str, value):
        self.root[key] = value

    def __contains__(self, key:str) -> bool:
        return key in self.root


linkml_meta = LinkMLMeta({'default_curi_maps': ['semweb_context'],
     'default_prefix': 'nexus',
     'default_range': 'string',
     'description': 'An ontology describing AI systems and their risks',
     'id': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology',
     'imports': ['linkml:types', 'common', 'ai_risk', 'ai_system', 'ai_eval'],
     'license': 'https://www.apache.org/licenses/LICENSE-2.0.html',
     'name': 'ai-risk-ontology',
     'prefixes': {'airo': {'prefix_prefix': 'airo',
                           'prefix_reference': 'https://w3id.org/airo#'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'nexus': {'prefix_prefix': 'nexus',
                            'prefix_reference': 'https://ibm.github.io/risk-atlas-nexus/ontology/'}},
     'source_file': 'src/risk_atlas_nexus/ai_risk_ontology/schema/ai-risk-ontology.yaml'} )

class EuAiRiskCategory(str, Enum):
    MINIMAL = "MINIMAL"
    LIMITED = "LIMITED"
    HIGH = "HIGH"
    UNACCEPTABLE = "UNACCEPTABLE"


class AiSystemType(str, Enum):
    # General-purpose AI (GPAI)
    GPAI = "GPAI"
    # General-purpose AI (GPAI) models released under free and open-source licences
    GPAI_OS = "GPAI_OS"
    # Prohibited AI system due to unacceptable risk category (e.g. social scoring systems and manipulative AI).
    PROHIBITED = "PROHIBITED"
    # AI used for scientific research and development
    SCIENTIFIC_RD = "SCIENTIFIC_RD"
    # AI used for military, defense and security purposes.
    MILITARY_SECURITY = "MILITARY_SECURITY"
    # AI systems pursuant to Article 6(1)(2) Classification Rules for High-Risk AI Systems
    HIGH_RISK = "HIGH_RISK"



class Entity(ConfiguredBaseModel):
    """
    A generic grouping for any identifiable entity.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'abstract': True,
         'class_uri': 'schema:Thing',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class Organization(Entity):
    """
    Any organizational entity such as a corporation, educational institution, consortium, government, etc.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'schema:Organization',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    grants_license: Optional[str] = Field(default=None, description="""A relationship from a granting entity such as an Organization to a License instance.""", json_schema_extra = { "linkml_meta": {'alias': 'grants_license', 'domain_of': ['Organization']} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class License(Entity):
    """
    The general notion of a license which defines terms and grants permissions to users of AI systems, datasets and software.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'airo:License',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    version: Optional[str] = Field(default=None, description="""The version of the entity embodied by a specified resource.""", json_schema_extra = { "linkml_meta": {'alias': 'version',
         'domain_of': ['License', 'RiskTaxonomy'],
         'slot_uri': 'schema:version'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class Dataset(Entity):
    """
    A body of structured information describing some topic(s) of interest.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'schema:Dataset',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset', 'RiskTaxonomy', 'AiEval', 'BaseAi'],
         'slot_uri': 'airo:hasLicense'} })
    hasDocumentation: Optional[List[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'RiskTaxonomy',
                       'Action',
                       'AiEval',
                       'BaseAi',
                       'LargeLanguageModelFamily'],
         'slot_uri': 'airo:hasDocumentation'} })
    provider: Optional[str] = Field(default=None, description="""A relationship to the Organization instance that provides this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'provider', 'domain_of': ['Dataset'], 'slot_uri': 'schema:provider'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class Documentation(Entity):
    """
    Documented information about a concept or other topic(s) of interest.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'airo:Documentation',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class Fact(ConfiguredBaseModel):
    """
    A fact about something, for example the result of a measurement. In addition to the value, evidence is provided.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'abstract': True,
         'class_uri': 'schema:Statement',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    value: str = Field(default=..., description="""Some numeric or string value""", json_schema_extra = { "linkml_meta": {'alias': 'value', 'domain_of': ['Fact']} })
    evidence: Optional[str] = Field(default=None, description="""Evidence provides a source (typical a chunk, paragraph or link) describing where some value was found or how it was generated.""", json_schema_extra = { "linkml_meta": {'alias': 'evidence', 'domain_of': ['Fact']} })


class RiskTaxonomy(Entity):
    """
    A taxonomy of AI system related risks
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    version: Optional[str] = Field(default=None, description="""The version of the entity embodied by a specified resource.""", json_schema_extra = { "linkml_meta": {'alias': 'version',
         'domain_of': ['License', 'RiskTaxonomy'],
         'slot_uri': 'schema:version'} })
    hasDocumentation: Optional[List[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'RiskTaxonomy',
                       'Action',
                       'AiEval',
                       'BaseAi',
                       'LargeLanguageModelFamily'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset', 'RiskTaxonomy', 'AiEval', 'BaseAi'],
         'slot_uri': 'airo:hasLicense'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class RiskConcept(Entity):
    """
    An umbrella term for refering to risk, risk source, consequence and impact.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'airo:RiskConcept',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    isDetectedBy: Optional[List[str]] = Field(default=None, description="""A relationship where a risk, risk source, consequence, or impact is detected by a risk control.""", json_schema_extra = { "linkml_meta": {'alias': 'isDetectedBy',
         'domain_of': ['RiskConcept'],
         'inverse': 'detectsRiskConcept'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class RiskGroup(RiskConcept, Entity):
    """
    A group of AI system related risks that are part of a risk taxonomy.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk',
         'mixins': ['RiskConcept'],
         'slot_usage': {'hasPart': {'description': 'A relationship where a riskgroup '
                                                   'has a risk',
                                    'name': 'hasPart',
                                    'range': 'Risk'}}})

    isDefinedByTaxonomy: Optional[str] = Field(default=None, description="""A relationship where a risk or a risk group is defined by a risk taxonomy""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByTaxonomy',
         'domain_of': ['RiskGroup', 'Risk', 'RiskControl', 'Action'],
         'slot_uri': 'schema:isPartOf'} })
    closeMatch: Optional[List[str]] = Field(default=None, description="""The property is used to link two concepts that are sufficiently similar that they can be used interchangeably in some information retrieval applications.""", json_schema_extra = { "linkml_meta": {'alias': 'closeMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:closeMatch'} })
    exactMatch: Optional[List[str]] = Field(default=None, description="""The property is used to link two concepts, indicating a high degree of confidence that the concepts can be used interchangeably across a wide range of information retrieval applications""", json_schema_extra = { "linkml_meta": {'alias': 'exactMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:exactMatch'} })
    broadMatch: Optional[List[str]] = Field(default=None, description="""The property is used to state a hierarchical mapping link between two concepts, indicating that the concept linked to, is a broader concept than the originating concept.""", json_schema_extra = { "linkml_meta": {'alias': 'broadMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:broadMatch'} })
    narrowMatch: Optional[List[str]] = Field(default=None, description="""The property is used to state a hierarchical mapping link between two concepts, indicating that the concept linked to, is a narrower concept than the originating concept.""", json_schema_extra = { "linkml_meta": {'alias': 'narrowMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:narrowMatch'} })
    relatedMatch: Optional[List[str]] = Field(default=None, description="""The property skos:relatedMatch is used to state an associative mapping link between two concepts.""", json_schema_extra = { "linkml_meta": {'alias': 'relatedMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:relatedMatch'} })
    hasPart: Optional[List[str]] = Field(default=None, description="""A relationship where a riskgroup has a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasPart', 'domain_of': ['RiskGroup'], 'slot_uri': 'schema:hasPart'} })
    isDetectedBy: Optional[List[str]] = Field(default=None, description="""A relationship where a risk, risk source, consequence, or impact is detected by a risk control.""", json_schema_extra = { "linkml_meta": {'alias': 'isDetectedBy',
         'domain_of': ['RiskConcept'],
         'inverse': 'detectsRiskConcept'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class Risk(RiskConcept, Entity):
    """
    The state of uncertainty associated with an AI system, that has the potential to cause harms
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'airo:Risk',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk',
         'mixins': ['RiskConcept'],
         'slot_usage': {'isPartOf': {'description': 'A relationship where a risk is '
                                                    'part of a risk group',
                                     'name': 'isPartOf',
                                     'range': 'RiskGroup'}}})

    hasRelatedAction: Optional[List[str]] = Field(default=None, description="""A relationship where an entity relates to an action""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedAction', 'domain_of': ['Risk']} })
    isDefinedByTaxonomy: Optional[str] = Field(default=None, description="""A relationship where a risk or a risk group is defined by a risk taxonomy""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByTaxonomy',
         'domain_of': ['RiskGroup', 'Risk', 'RiskControl', 'Action'],
         'slot_uri': 'schema:isPartOf'} })
    isPartOf: Optional[str] = Field(default=None, description="""A relationship where a risk is part of a risk group""", json_schema_extra = { "linkml_meta": {'alias': 'isPartOf',
         'domain_of': ['Risk', 'LargeLanguageModel'],
         'slot_uri': 'schema:isPartOf'} })
    closeMatch: Optional[List[str]] = Field(default=None, description="""The property is used to link two concepts that are sufficiently similar that they can be used interchangeably in some information retrieval applications.""", json_schema_extra = { "linkml_meta": {'alias': 'closeMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:closeMatch'} })
    exactMatch: Optional[List[str]] = Field(default=None, description="""The property is used to link two concepts, indicating a high degree of confidence that the concepts can be used interchangeably across a wide range of information retrieval applications""", json_schema_extra = { "linkml_meta": {'alias': 'exactMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:exactMatch'} })
    broadMatch: Optional[List[str]] = Field(default=None, description="""The property is used to state a hierarchical mapping link between two concepts, indicating that the concept linked to, is a broader concept than the originating concept.""", json_schema_extra = { "linkml_meta": {'alias': 'broadMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:broadMatch'} })
    narrowMatch: Optional[List[str]] = Field(default=None, description="""The property is used to state a hierarchical mapping link between two concepts, indicating that the concept linked to, is a narrower concept than the originating concept.""", json_schema_extra = { "linkml_meta": {'alias': 'narrowMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:narrowMatch'} })
    relatedMatch: Optional[List[str]] = Field(default=None, description="""The property skos:relatedMatch is used to state an associative mapping link between two concepts.""", json_schema_extra = { "linkml_meta": {'alias': 'relatedMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:relatedMatch'} })
    detectsRiskConcept: Optional[List[str]] = Field(default=None, description="""The property airo:detectsRiskConcept indicates the control used for detecting risks, risk sources,  consequences, and impacts.""", json_schema_extra = { "linkml_meta": {'alias': 'detectsRiskConcept',
         'domain_of': ['Risk', 'RiskControl'],
         'exact_mappings': ['airo:detectsRiskConcept'],
         'inverse': 'isDetectedBy'} })
    tag: Optional[str] = Field(default=None, description="""A shost version of the name""", json_schema_extra = { "linkml_meta": {'alias': 'tag', 'domain_of': ['Risk']} })
    type: Optional[str] = Field(default=None, description="""Annotation whether an AI risk occurs at input or output or is non-technical.""", json_schema_extra = { "linkml_meta": {'alias': 'type', 'domain_of': ['Risk']} })
    phase: Optional[str] = Field(default=None, description="""Annotation whether an AI risk shows specifically during the training-tuning or inference phase.""", json_schema_extra = { "linkml_meta": {'alias': 'phase', 'domain_of': ['Risk']} })
    descriptor: Optional[str] = Field(default=None, description="""Annotates whether an AI risk is a traditional risk, specific to or amplified by AI.""", json_schema_extra = { "linkml_meta": {'alias': 'descriptor', 'domain_of': ['Risk']} })
    concern: Optional[str] = Field(default=None, description="""Some explanation about the concern related to an AI risk""", json_schema_extra = { "linkml_meta": {'alias': 'concern', 'domain_of': ['Risk']} })
    isDetectedBy: Optional[List[str]] = Field(default=None, description="""A relationship where a risk, risk source, consequence, or impact is detected by a risk control.""", json_schema_extra = { "linkml_meta": {'alias': 'isDetectedBy',
         'domain_of': ['RiskConcept'],
         'inverse': 'detectsRiskConcept'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class RiskControl(Entity):
    """
    A measure that maintains and/or modifies risk (and risk concepts)
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'airo:RiskControl',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    detectsRiskConcept: Optional[List[str]] = Field(default=None, description="""The property airo:detectsRiskConcept indicates the control used for detecting risks, risk sources,  consequences, and impacts.""", json_schema_extra = { "linkml_meta": {'alias': 'detectsRiskConcept',
         'domain_of': ['Risk', 'RiskControl'],
         'exact_mappings': ['airo:detectsRiskConcept'],
         'inverse': 'isDetectedBy'} })
    isDefinedByTaxonomy: Optional[str] = Field(default=None, description="""A relationship where a risk or a risk group is defined by a risk taxonomy""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByTaxonomy',
         'domain_of': ['RiskGroup', 'Risk', 'RiskControl', 'Action'],
         'slot_uri': 'schema:isPartOf'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class Action(Entity):
    """
    Action to remediate a risk
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    hasRelatedRisk: Optional[List[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk', 'domain_of': ['Action', 'AiEval']} })
    hasDocumentation: Optional[List[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'RiskTaxonomy',
                       'Action',
                       'AiEval',
                       'BaseAi',
                       'LargeLanguageModelFamily'],
         'slot_uri': 'airo:hasDocumentation'} })
    isDefinedByTaxonomy: Optional[str] = Field(default=None, description="""A relationship where a risk or a risk group is defined by a risk taxonomy""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByTaxonomy',
         'domain_of': ['RiskGroup', 'Risk', 'RiskControl', 'Action'],
         'slot_uri': 'schema:isPartOf'} })
    hasAiActorTask: Optional[List[str]] = Field(default=None, description="""Pertinent AI Actor Tasks for each subcategory. Not every AI Actor Task listed will apply to every suggested action in the subcategory (i.e., some apply to AI development and others apply to AI deployment).""", json_schema_extra = { "linkml_meta": {'alias': 'hasAiActorTask', 'domain_of': ['Action']} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class AiEval(Entity):
    """
    An AI Evaluation, e.g. a metric, benchmark, unitxt card evaluation, a question or a combination of such entities.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dqv:Metric',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_eval',
         'slot_usage': {'isComposedOf': {'description': 'A relationship indicating '
                                                        'that an AI evaluation maybe '
                                                        'composed of other AI '
                                                        "evaluations (e.g. it's an "
                                                        'overall average of other '
                                                        'scores).',
                                         'name': 'isComposedOf',
                                         'range': 'AiEval'}}})

    hasDocumentation: Optional[List[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'RiskTaxonomy',
                       'Action',
                       'AiEval',
                       'BaseAi',
                       'LargeLanguageModelFamily'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasDataset: Optional[List[str]] = Field(default=None, description="""A relationship to datasets that are used.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDataset', 'domain_of': ['AiEval']} })
    hasUnitxtCard: Optional[str] = Field(default=None, description="""A relationship to a Unitxt card defining the risk evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'hasUnitxtCard', 'domain_of': ['AiEval'], 'slot_uri': 'schema:url'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset', 'RiskTaxonomy', 'AiEval', 'BaseAi'],
         'slot_uri': 'airo:hasLicense'} })
    hasRelatedRisk: Optional[List[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk', 'domain_of': ['Action', 'AiEval']} })
    bestValue: Optional[str] = Field(default=None, description="""Annotation of the best possible result of the evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'bestValue', 'domain_of': ['AiEval']} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class AiEvalResult(Fact, Entity):
    """
    The result of an evaluation for a specific AI model.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dqv:QualityMeasurement',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_eval',
         'mixins': ['Fact']})

    isResultOf: Optional[str] = Field(default=None, description="""A relationship indicating that an entity is the result of an AI evaluation.""", json_schema_extra = { "linkml_meta": {'alias': 'isResultOf',
         'domain_of': ['AiEvalResult'],
         'slot_uri': 'dqv:isMeasurementOf'} })
    value: str = Field(default=..., description="""Some numeric or string value""", json_schema_extra = { "linkml_meta": {'alias': 'value', 'domain_of': ['Fact']} })
    evidence: Optional[str] = Field(default=None, description="""Evidence provides a source (typical a chunk, paragraph or link) describing where some value was found or how it was generated.""", json_schema_extra = { "linkml_meta": {'alias': 'evidence', 'domain_of': ['Fact']} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class Question(AiEval):
    """
    An evaluation where a question has to be answered
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_eval'})

    text: str = Field(default=..., description="""The question itself""", json_schema_extra = { "linkml_meta": {'alias': 'text', 'domain_of': ['Question']} })
    hasDocumentation: Optional[List[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'RiskTaxonomy',
                       'Action',
                       'AiEval',
                       'BaseAi',
                       'LargeLanguageModelFamily'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasDataset: Optional[List[str]] = Field(default=None, description="""A relationship to datasets that are used.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDataset', 'domain_of': ['AiEval']} })
    hasUnitxtCard: Optional[str] = Field(default=None, description="""A relationship to a Unitxt card defining the risk evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'hasUnitxtCard', 'domain_of': ['AiEval'], 'slot_uri': 'schema:url'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset', 'RiskTaxonomy', 'AiEval', 'BaseAi'],
         'slot_uri': 'airo:hasLicense'} })
    hasRelatedRisk: Optional[List[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk', 'domain_of': ['Action', 'AiEval']} })
    bestValue: Optional[str] = Field(default=None, description="""Annotation of the best possible result of the evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'bestValue', 'domain_of': ['AiEval']} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class Questionnaire(AiEval):
    """
    A questionnaire groups questions
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_eval',
         'slot_usage': {'composed_of': {'name': 'composed_of', 'range': 'Question'}}})

    hasDocumentation: Optional[List[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'RiskTaxonomy',
                       'Action',
                       'AiEval',
                       'BaseAi',
                       'LargeLanguageModelFamily'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasDataset: Optional[List[str]] = Field(default=None, description="""A relationship to datasets that are used.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDataset', 'domain_of': ['AiEval']} })
    hasUnitxtCard: Optional[str] = Field(default=None, description="""A relationship to a Unitxt card defining the risk evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'hasUnitxtCard', 'domain_of': ['AiEval'], 'slot_uri': 'schema:url'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset', 'RiskTaxonomy', 'AiEval', 'BaseAi'],
         'slot_uri': 'airo:hasLicense'} })
    hasRelatedRisk: Optional[List[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk', 'domain_of': ['Action', 'AiEval']} })
    bestValue: Optional[str] = Field(default=None, description="""Annotation of the best possible result of the evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'bestValue', 'domain_of': ['AiEval']} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class AiOffice(Organization):
    """
    The EU AI Office (https://digital-strategy.ec.europa.eu/en/policies/ai-office)
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'schema:Organization',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/eu_ai_act'})

    grants_license: Optional[str] = Field(default=None, description="""A relationship from a granting entity such as an Organization to a License instance.""", json_schema_extra = { "linkml_meta": {'alias': 'grants_license', 'domain_of': ['Organization']} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class BaseAi(Entity):
    """
    Any type of AI, be it a LLM, RL agent, SVM, etc.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'abstract': True,
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system'})

    producer: Optional[str] = Field(default=None, description="""A relationship to the Organization instance which produces this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'producer', 'domain_of': ['BaseAi']} })
    hasModelCard: Optional[List[str]] = Field(default=None, description="""A relationship to model card references.""", json_schema_extra = { "linkml_meta": {'alias': 'hasModelCard', 'domain_of': ['BaseAi']} })
    hasDocumentation: Optional[List[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'RiskTaxonomy',
                       'Action',
                       'AiEval',
                       'BaseAi',
                       'LargeLanguageModelFamily'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset', 'RiskTaxonomy', 'AiEval', 'BaseAi'],
         'slot_uri': 'airo:hasLicense'} })
    performsTask: Optional[List[str]] = Field(default=None, description="""relationship indicating the AI tasks an AI model can perform.""", json_schema_extra = { "linkml_meta": {'alias': 'performsTask', 'domain_of': ['BaseAi']} })
    isProvidedBy: Optional[str] = Field(default=None, description="""A relationship indicating the AI model has been provided by an AI model provider.""", json_schema_extra = { "linkml_meta": {'alias': 'isProvidedBy',
         'domain_of': ['BaseAi'],
         'slot_uri': 'airo:isProvidedBy'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class AiSystem(BaseAi):
    """
    A compound AI System composed of one or more AI capablities. ChatGPT is an example of an AI system which deploys multiple GPT AI models.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'airo:AISystem',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system',
         'slot_usage': {'isComposedOf': {'description': 'Relationship indicating the '
                                                        'AI components from which a '
                                                        'complete AI system is '
                                                        'composed.',
                                         'name': 'isComposedOf',
                                         'range': 'BaseAi'}}})

    hasEuAiSystemType: Optional[AiSystemType] = Field(default=None, description="""The type of system as defined by the EU AI Act.""", json_schema_extra = { "linkml_meta": {'alias': 'hasEuAiSystemType', 'domain_of': ['AiSystem']} })
    hasEuRiskCategory: Optional[EuAiRiskCategory] = Field(default=None, description="""The risk category of an AI system as defined by the EU AI Act.""", json_schema_extra = { "linkml_meta": {'alias': 'hasEuRiskCategory', 'domain_of': ['AiSystem']} })
    producer: Optional[str] = Field(default=None, description="""A relationship to the Organization instance which produces this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'producer', 'domain_of': ['BaseAi']} })
    hasModelCard: Optional[List[str]] = Field(default=None, description="""A relationship to model card references.""", json_schema_extra = { "linkml_meta": {'alias': 'hasModelCard', 'domain_of': ['BaseAi']} })
    hasDocumentation: Optional[List[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'RiskTaxonomy',
                       'Action',
                       'AiEval',
                       'BaseAi',
                       'LargeLanguageModelFamily'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset', 'RiskTaxonomy', 'AiEval', 'BaseAi'],
         'slot_uri': 'airo:hasLicense'} })
    performsTask: Optional[List[str]] = Field(default=None, description="""relationship indicating the AI tasks an AI model can perform.""", json_schema_extra = { "linkml_meta": {'alias': 'performsTask', 'domain_of': ['BaseAi']} })
    isProvidedBy: Optional[str] = Field(default=None, description="""A relationship indicating the AI model has been provided by an AI model provider.""", json_schema_extra = { "linkml_meta": {'alias': 'isProvidedBy',
         'domain_of': ['BaseAi'],
         'slot_uri': 'airo:isProvidedBy'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class AiAgent(AiSystem):
    """
    An artificial intelligence (AI) agent refers to a system or program that is capable of autonomously performing tasks on behalf of a user or another system by designing its workflow and utilizing available tools.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system',
         'slot_usage': {'isProvidedBy': {'description': 'A relationship indicating the '
                                                        'AI agent has been provided by '
                                                        'an AI systems provider.',
                                         'name': 'isProvidedBy'}}})

    hasEuAiSystemType: Optional[AiSystemType] = Field(default=None, description="""The type of system as defined by the EU AI Act.""", json_schema_extra = { "linkml_meta": {'alias': 'hasEuAiSystemType', 'domain_of': ['AiSystem']} })
    hasEuRiskCategory: Optional[EuAiRiskCategory] = Field(default=None, description="""The risk category of an AI system as defined by the EU AI Act.""", json_schema_extra = { "linkml_meta": {'alias': 'hasEuRiskCategory', 'domain_of': ['AiSystem']} })
    producer: Optional[str] = Field(default=None, description="""A relationship to the Organization instance which produces this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'producer', 'domain_of': ['BaseAi']} })
    hasModelCard: Optional[List[str]] = Field(default=None, description="""A relationship to model card references.""", json_schema_extra = { "linkml_meta": {'alias': 'hasModelCard', 'domain_of': ['BaseAi']} })
    hasDocumentation: Optional[List[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'RiskTaxonomy',
                       'Action',
                       'AiEval',
                       'BaseAi',
                       'LargeLanguageModelFamily'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset', 'RiskTaxonomy', 'AiEval', 'BaseAi'],
         'slot_uri': 'airo:hasLicense'} })
    performsTask: Optional[List[str]] = Field(default=None, description="""relationship indicating the AI tasks an AI model can perform.""", json_schema_extra = { "linkml_meta": {'alias': 'performsTask', 'domain_of': ['BaseAi']} })
    isProvidedBy: Optional[str] = Field(default=None, description="""A relationship indicating the AI agent has been provided by an AI systems provider.""", json_schema_extra = { "linkml_meta": {'alias': 'isProvidedBy',
         'domain_of': ['BaseAi'],
         'slot_uri': 'airo:isProvidedBy'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class AiModel(BaseAi):
    """
    A base AI Model class. No assumption about the type (SVM, LLM, etc.). Subclassed by model types (see LargeLanguageModel).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system'})

    hasEvaluation: Optional[List[AiEvalResult]] = Field(default=None, description="""A relationship indicating that an entity has an AI evaluation result.""", json_schema_extra = { "linkml_meta": {'alias': 'hasEvaluation',
         'domain_of': ['AiModel'],
         'slot_uri': 'dqv:hasQualityMeasurement'} })
    architecture: Optional[str] = Field(default=None, description="""A description of the architecture of an AI such as 'Decoder-only'.""", json_schema_extra = { "linkml_meta": {'alias': 'architecture', 'domain_of': ['AiModel']} })
    gpu_hours: Optional[int] = Field(default=None, description="""GPU consumption in terms of hours""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'gpu_hours', 'domain_of': ['AiModel']} })
    power_consumption_w: Optional[int] = Field(default=None, description="""power consumption in Watts""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'power_consumption_w', 'domain_of': ['AiModel']} })
    carbon_emitted: Optional[float] = Field(default=None, description="""The number of tons of carbon dioxide equivalent that are emitted during training""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'carbon_emitted',
         'domain_of': ['AiModel'],
         'unit': {'descriptive_name': 'tons of CO2 equivalent', 'symbol': 't CO2-eq'}} })
    hasRiskControl: Optional[List[str]] = Field(default=None, description="""Indicates the control measures associated with a system or component to modify risks.""", json_schema_extra = { "linkml_meta": {'alias': 'hasRiskControl',
         'domain_of': ['AiModel'],
         'slot_uri': 'airo:hasRiskControl'} })
    producer: Optional[str] = Field(default=None, description="""A relationship to the Organization instance which produces this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'producer', 'domain_of': ['BaseAi']} })
    hasModelCard: Optional[List[str]] = Field(default=None, description="""A relationship to model card references.""", json_schema_extra = { "linkml_meta": {'alias': 'hasModelCard', 'domain_of': ['BaseAi']} })
    hasDocumentation: Optional[List[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'RiskTaxonomy',
                       'Action',
                       'AiEval',
                       'BaseAi',
                       'LargeLanguageModelFamily'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset', 'RiskTaxonomy', 'AiEval', 'BaseAi'],
         'slot_uri': 'airo:hasLicense'} })
    performsTask: Optional[List[str]] = Field(default=None, description="""relationship indicating the AI tasks an AI model can perform.""", json_schema_extra = { "linkml_meta": {'alias': 'performsTask', 'domain_of': ['BaseAi']} })
    isProvidedBy: Optional[str] = Field(default=None, description="""A relationship indicating the AI model has been provided by an AI model provider.""", json_schema_extra = { "linkml_meta": {'alias': 'isProvidedBy',
         'domain_of': ['BaseAi'],
         'slot_uri': 'airo:isProvidedBy'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class LargeLanguageModel(AiModel):
    """
    A large language model (LLM) is an AI model which supports a range of language-related tasks such as generation, summarization, classification, among others. A LLM is implemented as an artificial neural networks using a transformer architecture.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'aliases': ['LLM'],
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system',
         'slot_usage': {'isPartOf': {'description': 'Annotation that a Large Language '
                                                    'model is part of a family of '
                                                    'models',
                                     'name': 'isPartOf',
                                     'range': 'LargeLanguageModelFamily'}}})

    numParameters: Optional[int] = Field(default=None, description="""A property indicating the number of parameters in a LLM.""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'numParameters', 'domain_of': ['LargeLanguageModel']} })
    numTrainingTokens: Optional[int] = Field(default=None, description="""The number of tokens a AI model was trained on.""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'numTrainingTokens', 'domain_of': ['LargeLanguageModel']} })
    contextWindowSize: Optional[int] = Field(default=None, description="""The total length, in bytes, of an AI model's context window.""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'contextWindowSize', 'domain_of': ['LargeLanguageModel']} })
    hasInputModality: Optional[List[str]] = Field(default=None, description="""A relationship indicating the input modalities supported by an AI component. Examples include text, image, video.""", json_schema_extra = { "linkml_meta": {'alias': 'hasInputModality', 'domain_of': ['LargeLanguageModel']} })
    hasOutputModality: Optional[List[str]] = Field(default=None, description="""A relationship indicating the output modalities supported by an AI component. Examples include text, image, video.""", json_schema_extra = { "linkml_meta": {'alias': 'hasOutputModality', 'domain_of': ['LargeLanguageModel']} })
    hasTrainingData: Optional[List[str]] = Field(default=None, description="""A relationship indicating the datasets an AI model was trained on.""", json_schema_extra = { "linkml_meta": {'alias': 'hasTrainingData',
         'domain_of': ['LargeLanguageModel'],
         'slot_uri': 'airo:hasTrainingData'} })
    fine_tuning: Optional[str] = Field(default=None, description="""A description of the fine-tuning mechanism(s) applied to a model.""", json_schema_extra = { "linkml_meta": {'alias': 'fine_tuning', 'domain_of': ['LargeLanguageModel']} })
    supported_languages: Optional[List[str]] = Field(default=None, description="""A list of languages, expressed as ISO two letter codes. For example, 'jp, fr, en, de'""", json_schema_extra = { "linkml_meta": {'alias': 'supported_languages', 'domain_of': ['LargeLanguageModel']} })
    isPartOf: Optional[str] = Field(default=None, description="""Annotation that a Large Language model is part of a family of models""", json_schema_extra = { "linkml_meta": {'alias': 'isPartOf',
         'domain_of': ['Risk', 'LargeLanguageModel'],
         'slot_uri': 'schema:isPartOf'} })
    hasEvaluation: Optional[List[AiEvalResult]] = Field(default=None, description="""A relationship indicating that an entity has an AI evaluation result.""", json_schema_extra = { "linkml_meta": {'alias': 'hasEvaluation',
         'domain_of': ['AiModel'],
         'slot_uri': 'dqv:hasQualityMeasurement'} })
    architecture: Optional[str] = Field(default=None, description="""A description of the architecture of an AI such as 'Decoder-only'.""", json_schema_extra = { "linkml_meta": {'alias': 'architecture', 'domain_of': ['AiModel']} })
    gpu_hours: Optional[int] = Field(default=None, description="""GPU consumption in terms of hours""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'gpu_hours', 'domain_of': ['AiModel']} })
    power_consumption_w: Optional[int] = Field(default=None, description="""power consumption in Watts""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'power_consumption_w', 'domain_of': ['AiModel']} })
    carbon_emitted: Optional[float] = Field(default=None, description="""The number of tons of carbon dioxide equivalent that are emitted during training""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'carbon_emitted',
         'domain_of': ['AiModel'],
         'unit': {'descriptive_name': 'tons of CO2 equivalent', 'symbol': 't CO2-eq'}} })
    hasRiskControl: Optional[List[str]] = Field(default=None, description="""Indicates the control measures associated with a system or component to modify risks.""", json_schema_extra = { "linkml_meta": {'alias': 'hasRiskControl',
         'domain_of': ['AiModel'],
         'slot_uri': 'airo:hasRiskControl'} })
    producer: Optional[str] = Field(default=None, description="""A relationship to the Organization instance which produces this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'producer', 'domain_of': ['BaseAi']} })
    hasModelCard: Optional[List[str]] = Field(default=None, description="""A relationship to model card references.""", json_schema_extra = { "linkml_meta": {'alias': 'hasModelCard', 'domain_of': ['BaseAi']} })
    hasDocumentation: Optional[List[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'RiskTaxonomy',
                       'Action',
                       'AiEval',
                       'BaseAi',
                       'LargeLanguageModelFamily'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset', 'RiskTaxonomy', 'AiEval', 'BaseAi'],
         'slot_uri': 'airo:hasLicense'} })
    performsTask: Optional[List[str]] = Field(default=None, description="""relationship indicating the AI tasks an AI model can perform.""", json_schema_extra = { "linkml_meta": {'alias': 'performsTask', 'domain_of': ['BaseAi']} })
    isProvidedBy: Optional[str] = Field(default=None, description="""A relationship indicating the AI model has been provided by an AI model provider.""", json_schema_extra = { "linkml_meta": {'alias': 'isProvidedBy',
         'domain_of': ['BaseAi'],
         'slot_uri': 'airo:isProvidedBy'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class LargeLanguageModelFamily(Entity):
    """
    A large language model family is a set of models that are provided by the same AI systems provider and are built around the same architecture, but differ e.g. in the number of parameters. Examples are Meta's Llama2 family or the IBM granite models.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system'})

    hasDocumentation: Optional[List[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'RiskTaxonomy',
                       'Action',
                       'AiEval',
                       'BaseAi',
                       'LargeLanguageModelFamily'],
         'slot_uri': 'airo:hasDocumentation'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class AiTask(Entity):
    """
    A task, such as summarization and classification, performed by an AI.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'airo:AiCapability',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class AiLifecyclePhase(Entity):
    """
    A Phase of AI lifecycle which indicates evolution of the system from conception through retirement.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'abstract': True,
         'class_uri': 'airo:AILifecyclePhase',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class DataPreprocessing(AiLifecyclePhase):
    """
    Data transformations, such as PI filtering, performed to ensure high quality of AI model training data.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class AiModelValidation(AiLifecyclePhase):
    """
    AI model validation steps that have been performed after the model training to ensure high AI model quality.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class AiProvider(Organization):
    """
    A provider under the AI Act is defined by Article 3(3) as a natural or legal person or body that develops an AI system or general-purpose AI model or has an AI system or general-purpose AI model developed; and places that system or model on the market, or puts that system into service, under the provider's own name or trademark, whether for payment or free for charge.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'airo:AIProvider',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system'})

    grants_license: Optional[str] = Field(default=None, description="""A relationship from a granting entity such as an Organization to a License instance.""", json_schema_extra = { "linkml_meta": {'alias': 'grants_license', 'domain_of': ['Organization']} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class Modality(Entity):
    """
    A modality supported by an Ai component. Examples include text, image, video.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'airo:Modality',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity'], 'slot_uri': 'schema:name'} })
    description: Optional[str] = Field(default=None, description="""The description of an entity""", json_schema_extra = { "linkml_meta": {'alias': 'description',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:description'} })
    url: Optional[str] = Field(default=None, description="""An optional URL associated with this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'url', 'domain_of': ['Entity'], 'slot_uri': 'schema:url'} })
    dateCreated: Optional[date] = Field(default=None, description="""The date on which the entity was created.""", json_schema_extra = { "linkml_meta": {'alias': 'dateCreated',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateCreated'} })
    dateModified: Optional[date] = Field(default=None, description="""The date on which the entity was most recently modified.""", json_schema_extra = { "linkml_meta": {'alias': 'dateModified',
         'domain_of': ['Entity'],
         'slot_uri': 'schema:dateModified'} })


class Container(ConfiguredBaseModel):
    """
    An umbrella object that holds the ontology class instances
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai-risk-ontology',
         'tree_root': True})

    organizations: Optional[List[Organization]] = Field(default=None, description="""A list of organizations""", json_schema_extra = { "linkml_meta": {'alias': 'organizations', 'domain_of': ['Container']} })
    licenses: Optional[List[License]] = Field(default=None, description="""A list of licenses""", json_schema_extra = { "linkml_meta": {'alias': 'licenses', 'domain_of': ['Container']} })
    modalities: Optional[List[Modality]] = Field(default=None, description="""A list of AI modalities""", json_schema_extra = { "linkml_meta": {'alias': 'modalities', 'domain_of': ['Container']} })
    aitasks: Optional[List[AiTask]] = Field(default=None, description="""A list of AI tasks""", json_schema_extra = { "linkml_meta": {'alias': 'aitasks', 'domain_of': ['Container']} })
    documents: Optional[List[Documentation]] = Field(default=None, description="""A list of documents""", json_schema_extra = { "linkml_meta": {'alias': 'documents', 'domain_of': ['Container']} })
    datasets: Optional[List[Dataset]] = Field(default=None, description="""A list of data sets""", json_schema_extra = { "linkml_meta": {'alias': 'datasets', 'domain_of': ['Container']} })
    taxonomies: Optional[List[RiskTaxonomy]] = Field(default=None, description="""A list of AI risk taxonomies""", json_schema_extra = { "linkml_meta": {'alias': 'taxonomies', 'domain_of': ['Container']} })
    riskgroups: Optional[List[RiskGroup]] = Field(default=None, description="""A list of AI risk groups""", json_schema_extra = { "linkml_meta": {'alias': 'riskgroups', 'domain_of': ['Container']} })
    risks: Optional[List[Risk]] = Field(default=None, description="""A list of AI risks""", json_schema_extra = { "linkml_meta": {'alias': 'risks', 'domain_of': ['Container']} })
    riskcontrols: Optional[List[RiskControl]] = Field(default=None, description="""A list of AI risk controls""", json_schema_extra = { "linkml_meta": {'alias': 'riskcontrols', 'domain_of': ['Container']} })
    actions: Optional[List[Action]] = Field(default=None, description="""A list of risk related actions""", json_schema_extra = { "linkml_meta": {'alias': 'actions', 'domain_of': ['Container']} })
    evaluations: Optional[List[AiEval]] = Field(default=None, description="""A list of AI evaluation methods""", json_schema_extra = { "linkml_meta": {'alias': 'evaluations', 'domain_of': ['Container']} })
    aimodelfamilies: Optional[List[LargeLanguageModelFamily]] = Field(default=None, description="""A list of AI model families""", json_schema_extra = { "linkml_meta": {'alias': 'aimodelfamilies', 'domain_of': ['Container']} })
    aimodels: Optional[List[LargeLanguageModel]] = Field(default=None, description="""A list of AI models""", json_schema_extra = { "linkml_meta": {'alias': 'aimodels', 'domain_of': ['Container']} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
Entity.model_rebuild()
Organization.model_rebuild()
License.model_rebuild()
Dataset.model_rebuild()
Documentation.model_rebuild()
Fact.model_rebuild()
RiskTaxonomy.model_rebuild()
RiskConcept.model_rebuild()
RiskGroup.model_rebuild()
Risk.model_rebuild()
RiskControl.model_rebuild()
Action.model_rebuild()
AiEval.model_rebuild()
AiEvalResult.model_rebuild()
Question.model_rebuild()
Questionnaire.model_rebuild()
AiOffice.model_rebuild()
BaseAi.model_rebuild()
AiSystem.model_rebuild()
AiAgent.model_rebuild()
AiModel.model_rebuild()
LargeLanguageModel.model_rebuild()
LargeLanguageModelFamily.model_rebuild()
AiTask.model_rebuild()
AiLifecyclePhase.model_rebuild()
DataPreprocessing.model_rebuild()
AiModelValidation.model_rebuild()
AiProvider.model_rebuild()
Modality.model_rebuild()
Container.model_rebuild()

