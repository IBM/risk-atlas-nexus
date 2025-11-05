from __future__ import annotations

import re
import sys
from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
from typing import Any, ClassVar, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, RootModel, field_validator


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
    root: dict[str, Any] = {}
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
     'imports': ['linkml:types',
                 'common',
                 'ai_risk',
                 'ai_system',
                 'ai_eval',
                 'ai_intrinsic',
                 'ai_csiro_rai'],
     'license': 'https://www.apache.org/licenses/LICENSE-2.0.html',
     'name': 'ai-risk-ontology',
     'prefixes': {'airo': {'prefix_prefix': 'airo',
                           'prefix_reference': 'https://w3id.org/airo#'},
                  'linkml': {'prefix_prefix': 'linkml',
                             'prefix_reference': 'https://w3id.org/linkml/'},
                  'nexus': {'prefix_prefix': 'nexus',
                            'prefix_reference': 'https://ibm.github.io/risk-atlas-nexus/ontology/'}},
     'source_file': 'src/risk_atlas_nexus/ai_risk_ontology/schema/ai-risk-ontology.yaml'} )

class AdapterType(str, Enum):
    LORA = "LORA"
    """
    Low-rank adapters, or LoRAs, are a fast way to give generalist large language models targeted knowledge and skills so they can do things like summarize IT manuals or rate the accuracy of their own answers. LoRA reduces the number of trainable parameters by learning pairs of rank-decompostion matrices while freezing the original weights. This vastly reduces the storage requirement for large language models adapted to specific tasks and enables efficient task-switching during deployment all without introducing inference latency. LoRA also outperforms several other adaptation methods including adapter, prefix-tuning, and fine-tuning. See arXiv:2106.09685
    """
    ALORA = "ALORA"
    """
    Activated LoRA (aLoRA) is a low rank adapter architecture that allows for reusing existing base model KV cache for more efficient inference, unlike standard LoRA models. As a result, aLoRA models can be quickly invoked as-needed for specialized tasks during (long) flows where the base model is primarily used, avoiding potentially expensive prefill costs in terms of latency, throughput, and GPU memory. See arXiv:2504.12397 for further details.
    """
    X_LORA = "X-LORA"
    """
    Mixture of LoRA Experts (X-LoRA) is a mixture of experts method for LoRA which works by using dense or sparse gating to dynamically activate LoRA experts.
    """


class EuAiRiskCategory(str, Enum):
    EXCLUDED = "EXCLUDED"
    """
    Excluded
    """
    PROHIBITED = "PROHIBITED"
    """
    Prohibited
    """
    HIGH_RISK_EXCEPTION = "HIGH_RISK_EXCEPTION"
    """
    High-Risk Exception
    """
    HIGH_RISK = "HIGH_RISK"
    """
    High Risk
    """
    LIMITED_OR_LOW_RISK = "LIMITED_OR_LOW_RISK"
    """
    Limited or Low Risk
    """


class AiSystemType(str, Enum):
    GPAI = "GPAI"
    """
    General-purpose AI (GPAI)
    """
    GPAI_OS = "GPAI_OS"
    """
    General-purpose AI (GPAI) models released under free and open-source licences
    """
    PROHIBITED = "PROHIBITED"
    """
    Prohibited AI system due to unacceptable risk category (e.g. social scoring systems and manipulative AI).
    """
    SCIENTIFIC_RD = "SCIENTIFIC_RD"
    """
    AI used for scientific research and development
    """
    MILITARY_SECURITY = "MILITARY_SECURITY"
    """
    AI used for military, defense and security purposes.
    """
    HIGH_RISK = "HIGH_RISK"
    """
    AI systems pursuant to Article 6(1)(2) Classification Rules for High-Risk AI Systems
    """



class Entity(ConfiguredBaseModel):
    """
    A generic grouping for any identifiable entity.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'abstract': True,
         'class_uri': 'schema:Thing',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
         'domain_of': ['License', 'Vocabulary', 'RiskTaxonomy'],
         'slot_uri': 'schema:version'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    provider: Optional[str] = Field(default=None, description="""A relationship to the Organization instance that provides this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'provider', 'domain_of': ['Dataset'], 'slot_uri': 'schema:provider'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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

    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    author: Optional[str] = Field(default=None, description="""The author or authors of the documentation""", json_schema_extra = { "linkml_meta": {'alias': 'author', 'domain_of': ['Documentation', 'RiskIncident']} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Vocabulary(Entity):
    """
    A collection of terms, with their definitions and relationships.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    version: Optional[str] = Field(default=None, description="""The version of the entity embodied by a specified resource.""", json_schema_extra = { "linkml_meta": {'alias': 'version',
         'domain_of': ['License', 'Vocabulary', 'RiskTaxonomy'],
         'slot_uri': 'schema:version'} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Term(Entity):
    """
    A term and its definitions
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    isDefinedByVocabulary: Optional[str] = Field(default=None, description="""A relationship where a term or a term group is defined by a vocabulary""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByVocabulary',
         'domain_of': ['Term', 'Adapter', 'LLMIntrinsic'],
         'slot_uri': 'schema:isPartOf'} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasParentDefinition: Optional[list[str]] = Field(default=None, description="""Indicates parent terms associated with a term""", json_schema_extra = { "linkml_meta": {'alias': 'hasParentDefinition',
         'domain_of': ['Term'],
         'slot_uri': 'nexus:hasParentDefinition'} })
    hasSubDefinition: Optional[list[str]] = Field(default=None, description="""Indicates child terms associated with a term""", json_schema_extra = { "linkml_meta": {'alias': 'hasSubDefinition',
         'domain_of': ['Term'],
         'slot_uri': 'nexus:hasSubDefinition'} })
    hasRelatedRisk: Optional[list[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk',
         'any_of': [{'range': 'RiskConcept'}, {'range': 'Term'}],
         'domain': 'Any',
         'domain_of': ['Term',
                       'LLMQuestionPolicy',
                       'Action',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic']} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Principle(Entity):
    """
    A representation of values or norms that must be taken into consideration when conducting activities
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:Principle',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Policy(Entity):
    """
    A guidance document outlining any of: procedures, plans, principles, decisions, intent, or protocols.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:Policy',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    isDefinedByTaxonomy: Optional[str] = Field(default=None, description="""A relationship where a risk or a risk group is defined by a risk taxonomy""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByTaxonomy',
         'domain_of': ['Policy',
                       'RiskGroup',
                       'Risk',
                       'RiskControl',
                       'Action',
                       'RiskIncident',
                       'StakeholderGroup',
                       'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class LLMQuestionPolicy(Policy):
    """
    The policy guides how the language model should answer a diverse set of sensitive questions.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    hasRelatedRisk: Optional[list[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk',
         'any_of': [{'range': 'RiskConcept'}, {'range': 'Term'}],
         'domain': 'Any',
         'domain_of': ['Term',
                       'LLMQuestionPolicy',
                       'Action',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic']} })
    hasRule: Optional[list[str]] = Field(default=None, description="""Specifying applicability or inclusion of a rule within specified context.""", json_schema_extra = { "linkml_meta": {'alias': 'hasRule',
         'domain_of': ['LLMQuestionPolicy'],
         'slot_uri': 'dpv:hasRule'} })
    hasReasonDenial: Optional[str] = Field(default=None, description="""Reason for denial""", json_schema_extra = { "linkml_meta": {'alias': 'hasReasonDenial',
         'domain_of': ['LLMQuestionPolicy'],
         'slot_uri': 'nexus:hasReasonDenial'} })
    hasShortReplyType: Optional[str] = Field(default=None, description="""Short reply type""", json_schema_extra = { "linkml_meta": {'alias': 'hasShortReplyType',
         'domain_of': ['LLMQuestionPolicy'],
         'slot_uri': 'nexus:hasShortReplyType'} })
    hasException: Optional[str] = Field(default=None, description="""Exception type""", json_schema_extra = { "linkml_meta": {'alias': 'hasException',
         'domain_of': ['LLMQuestionPolicy'],
         'slot_uri': 'nexus:hasException'} })
    isDefinedByTaxonomy: Optional[str] = Field(default=None, description="""A relationship where a risk or a risk group is defined by a risk taxonomy""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByTaxonomy',
         'domain_of': ['Policy',
                       'RiskGroup',
                       'Risk',
                       'RiskControl',
                       'Action',
                       'RiskIncident',
                       'StakeholderGroup',
                       'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Rule(Entity):
    """
    A rule describing a process or control that directs or determines if and how an activity should be conducted.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:Rule',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Permission(Rule):
    """
    A rule describing a permission to perform an activity
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:Permission',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Prohibition(Rule):
    """
    A rule describing a prohibition to perform an activity
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:Prohibition',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Obligation(Rule):
    """
    A rule describing an obligation for performing an activity
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:Obligation',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/common'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class RiskTaxonomy(Entity):
    """
    A taxonomy of AI system related risks
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    version: Optional[str] = Field(default=None, description="""The version of the entity embodied by a specified resource.""", json_schema_extra = { "linkml_meta": {'alias': 'version',
         'domain_of': ['License', 'Vocabulary', 'RiskTaxonomy'],
         'slot_uri': 'schema:version'} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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

    isDetectedBy: Optional[list[str]] = Field(default=None, description="""A relationship where a risk, risk source, consequence, or impact is detected by a risk control.""", json_schema_extra = { "linkml_meta": {'alias': 'isDetectedBy',
         'domain': 'RiskConcept',
         'domain_of': ['RiskConcept'],
         'inverse': 'detectsRiskConcept'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
         'domain_of': ['Policy',
                       'RiskGroup',
                       'Risk',
                       'RiskControl',
                       'Action',
                       'RiskIncident',
                       'StakeholderGroup',
                       'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    closeMatch: Optional[list[str]] = Field(default=None, description="""The property is used to link two concepts that are sufficiently similar that they can be used interchangeably in some information retrieval applications.""", json_schema_extra = { "linkml_meta": {'alias': 'closeMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:closeMatch'} })
    exactMatch: Optional[list[str]] = Field(default=None, description="""The property is used to link two concepts, indicating a high degree of confidence that the concepts can be used interchangeably across a wide range of information retrieval applications""", json_schema_extra = { "linkml_meta": {'alias': 'exactMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:exactMatch'} })
    broadMatch: Optional[list[str]] = Field(default=None, description="""The property is used to state a hierarchical mapping link between two concepts, indicating that the concept linked to, is a broader concept than the originating concept.""", json_schema_extra = { "linkml_meta": {'alias': 'broadMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:broadMatch'} })
    narrowMatch: Optional[list[str]] = Field(default=None, description="""The property is used to state a hierarchical mapping link between two concepts, indicating that the concept linked to, is a narrower concept than the originating concept.""", json_schema_extra = { "linkml_meta": {'alias': 'narrowMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:narrowMatch'} })
    relatedMatch: Optional[list[str]] = Field(default=None, description="""The property skos:relatedMatch is used to state an associative mapping link between two concepts.""", json_schema_extra = { "linkml_meta": {'alias': 'relatedMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:relatedMatch'} })
    hasPart: Optional[list[str]] = Field(default=None, description="""A relationship where a riskgroup has a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasPart', 'domain_of': ['RiskGroup'], 'slot_uri': 'schema:hasPart'} })
    isDetectedBy: Optional[list[str]] = Field(default=None, description="""A relationship where a risk, risk source, consequence, or impact is detected by a risk control.""", json_schema_extra = { "linkml_meta": {'alias': 'isDetectedBy',
         'domain': 'RiskConcept',
         'domain_of': ['RiskConcept'],
         'inverse': 'detectsRiskConcept'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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

    hasRelatedAction: Optional[list[str]] = Field(default=None, description="""A relationship where an entity relates to an action""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedAction', 'domain_of': ['Risk']} })
    isDefinedByTaxonomy: Optional[str] = Field(default=None, description="""A relationship where a risk or a risk group is defined by a risk taxonomy""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByTaxonomy',
         'domain_of': ['Policy',
                       'RiskGroup',
                       'Risk',
                       'RiskControl',
                       'Action',
                       'RiskIncident',
                       'StakeholderGroup',
                       'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    isPartOf: Optional[str] = Field(default=None, description="""A relationship where a risk is part of a risk group""", json_schema_extra = { "linkml_meta": {'alias': 'isPartOf',
         'domain_of': ['Risk', 'LargeLanguageModel', 'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    closeMatch: Optional[list[str]] = Field(default=None, description="""The property is used to link two concepts that are sufficiently similar that they can be used interchangeably in some information retrieval applications.""", json_schema_extra = { "linkml_meta": {'alias': 'closeMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:closeMatch'} })
    exactMatch: Optional[list[str]] = Field(default=None, description="""The property is used to link two concepts, indicating a high degree of confidence that the concepts can be used interchangeably across a wide range of information retrieval applications""", json_schema_extra = { "linkml_meta": {'alias': 'exactMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:exactMatch'} })
    broadMatch: Optional[list[str]] = Field(default=None, description="""The property is used to state a hierarchical mapping link between two concepts, indicating that the concept linked to, is a broader concept than the originating concept.""", json_schema_extra = { "linkml_meta": {'alias': 'broadMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:broadMatch'} })
    narrowMatch: Optional[list[str]] = Field(default=None, description="""The property is used to state a hierarchical mapping link between two concepts, indicating that the concept linked to, is a narrower concept than the originating concept.""", json_schema_extra = { "linkml_meta": {'alias': 'narrowMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:narrowMatch'} })
    relatedMatch: Optional[list[str]] = Field(default=None, description="""The property skos:relatedMatch is used to state an associative mapping link between two concepts.""", json_schema_extra = { "linkml_meta": {'alias': 'relatedMatch',
         'any_of': [{'range': 'Risk'}, {'range': 'RiskGroup'}],
         'domain_of': ['RiskGroup', 'Risk'],
         'slot_uri': 'skos:relatedMatch'} })
    detectsRiskConcept: Optional[list[str]] = Field(default=None, description="""The property airo:detectsRiskConcept indicates the control used for detecting risks, risk sources, consequences, and impacts.""", json_schema_extra = { "linkml_meta": {'alias': 'detectsRiskConcept',
         'domain': 'RiskControl',
         'domain_of': ['Risk', 'RiskControl'],
         'exact_mappings': ['airo:detectsRiskConcept'],
         'inverse': 'isDetectedBy'} })
    tag: Optional[str] = Field(default=None, description="""A shost version of the name""", json_schema_extra = { "linkml_meta": {'alias': 'tag', 'domain_of': ['Risk']} })
    type: Optional[str] = Field(default=None, description="""Annotation whether an AI risk occurs at input or output or is non-technical.""", json_schema_extra = { "linkml_meta": {'alias': 'type', 'domain_of': ['Risk']} })
    phase: Optional[str] = Field(default=None, description="""Annotation whether an AI risk shows specifically during the training-tuning or inference phase.""", json_schema_extra = { "linkml_meta": {'alias': 'phase', 'domain_of': ['Risk']} })
    descriptor: Optional[list[str]] = Field(default=None, description="""Annotates whether an AI risk is a traditional risk, specific to or amplified by AI.""", json_schema_extra = { "linkml_meta": {'alias': 'descriptor', 'domain_of': ['Risk']} })
    concern: Optional[str] = Field(default=None, description="""Some explanation about the concern related to an AI risk""", json_schema_extra = { "linkml_meta": {'alias': 'concern', 'domain_of': ['Risk']} })
    isDetectedBy: Optional[list[str]] = Field(default=None, description="""A relationship where a risk, risk source, consequence, or impact is detected by a risk control.""", json_schema_extra = { "linkml_meta": {'alias': 'isDetectedBy',
         'domain': 'RiskConcept',
         'domain_of': ['RiskConcept'],
         'inverse': 'detectsRiskConcept'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class RiskControl(RiskConcept, Entity):
    """
    A measure that maintains and/or modifies risk (and risk concepts)
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'airo:RiskControl',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk',
         'mixins': ['RiskConcept']})

    detectsRiskConcept: Optional[list[str]] = Field(default=None, description="""The property airo:detectsRiskConcept indicates the control used for detecting risks, risk sources, consequences, and impacts.""", json_schema_extra = { "linkml_meta": {'alias': 'detectsRiskConcept',
         'domain': 'RiskControl',
         'domain_of': ['Risk', 'RiskControl'],
         'exact_mappings': ['airo:detectsRiskConcept'],
         'inverse': 'isDetectedBy'} })
    isDefinedByTaxonomy: Optional[str] = Field(default=None, description="""A relationship where a risk or a risk group is defined by a risk taxonomy""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByTaxonomy',
         'domain_of': ['Policy',
                       'RiskGroup',
                       'Risk',
                       'RiskControl',
                       'Action',
                       'RiskIncident',
                       'StakeholderGroup',
                       'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    isDetectedBy: Optional[list[str]] = Field(default=None, description="""A relationship where a risk, risk source, consequence, or impact is detected by a risk control.""", json_schema_extra = { "linkml_meta": {'alias': 'isDetectedBy',
         'domain': 'RiskConcept',
         'domain_of': ['RiskConcept'],
         'inverse': 'detectsRiskConcept'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Action(RiskControl):
    """
    Action to remediate a risk
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    hasRelatedRisk: Optional[list[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk',
         'any_of': [{'range': 'RiskConcept'}, {'range': 'Term'}],
         'domain': 'Any',
         'domain_of': ['Term',
                       'LLMQuestionPolicy',
                       'Action',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic']} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    isDefinedByTaxonomy: Optional[str] = Field(default=None, description="""A relationship where a risk or a risk group is defined by a risk taxonomy""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByTaxonomy',
         'domain_of': ['Policy',
                       'RiskGroup',
                       'Risk',
                       'RiskControl',
                       'Action',
                       'RiskIncident',
                       'StakeholderGroup',
                       'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    hasAiActorTask: Optional[list[str]] = Field(default=None, description="""Pertinent AI Actor Tasks for each subcategory. Not every AI Actor Task listed will apply to every suggested action in the subcategory (i.e., some apply to AI development and others apply to AI deployment).""", json_schema_extra = { "linkml_meta": {'alias': 'hasAiActorTask', 'domain_of': ['Action']} })
    detectsRiskConcept: Optional[list[str]] = Field(default=None, description="""The property airo:detectsRiskConcept indicates the control used for detecting risks, risk sources, consequences, and impacts.""", json_schema_extra = { "linkml_meta": {'alias': 'detectsRiskConcept',
         'domain': 'RiskControl',
         'domain_of': ['Risk', 'RiskControl'],
         'exact_mappings': ['airo:detectsRiskConcept'],
         'inverse': 'isDetectedBy'} })
    isDetectedBy: Optional[list[str]] = Field(default=None, description="""A relationship where a risk, risk source, consequence, or impact is detected by a risk control.""", json_schema_extra = { "linkml_meta": {'alias': 'isDetectedBy',
         'domain': 'RiskConcept',
         'domain_of': ['RiskConcept'],
         'inverse': 'detectsRiskConcept'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class RiskIncident(RiskConcept, Entity):
    """
    An event occuring or occured which is a realised or materialised risk.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'https://w3id.org/dpv/risk#Incident',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk',
         'mixins': ['RiskConcept']})

    refersToRisk: Optional[list[str]] = Field(default=None, description="""Indicates the incident (subject) is a materialisation of the indicated risk (object)""", json_schema_extra = { "linkml_meta": {'alias': 'refersToRisk',
         'domain': 'RiskIncident',
         'domain_of': ['RiskIncident'],
         'exact_mappings': ['dpv:refersToRisk']} })
    isDefinedByTaxonomy: Optional[str] = Field(default=None, description="""A relationship where a risk or a risk group is defined by a risk taxonomy""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByTaxonomy',
         'domain_of': ['Policy',
                       'RiskGroup',
                       'Risk',
                       'RiskControl',
                       'Action',
                       'RiskIncident',
                       'StakeholderGroup',
                       'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    hasStatus: Optional[str] = Field(default=None, description="""Indicates the status of specified concept""", json_schema_extra = { "linkml_meta": {'alias': 'hasStatus', 'domain': 'RiskConcept', 'domain_of': ['RiskIncident']} })
    hasSeverity: Optional[str] = Field(default=None, description="""Indicates the severity associated with a concept""", json_schema_extra = { "linkml_meta": {'alias': 'hasSeverity', 'domain': 'RiskConcept', 'domain_of': ['RiskIncident']} })
    hasLikelihood: Optional[str] = Field(default=None, description="""The likelihood or probability or chance of something taking place or occuring""", json_schema_extra = { "linkml_meta": {'alias': 'hasLikelihood',
         'domain': 'RiskConcept',
         'domain_of': ['RiskIncident']} })
    hasImpactOn: Optional[str] = Field(default=None, description="""Indicates impact(s) possible or arising as consequences from specified concept""", json_schema_extra = { "linkml_meta": {'alias': 'hasImpactOn',
         'broad_mappings': ['dpv:hasConsequenceOn'],
         'domain': 'RiskConcept',
         'domain_of': ['RiskIncident']} })
    hasConsequence: Optional[str] = Field(default=None, description="""Indicates consequence(s) possible or arising from specified concept""", json_schema_extra = { "linkml_meta": {'alias': 'hasConsequence',
         'domain': 'RiskConcept',
         'domain_of': ['RiskIncident']} })
    hasImpact: Optional[str] = Field(default=None, description="""Indicates impact(s) possible or arising as consequences from specified concept""", json_schema_extra = { "linkml_meta": {'alias': 'hasImpact',
         'broad_mappings': ['dpv:hasConsequence'],
         'domain': 'RiskConcept',
         'domain_of': ['RiskIncident']} })
    hasVariant: Optional[str] = Field(default=None, description="""Indicates an incident that shares the same causative factors, produces similar harms, and involves the same intelligent systems as a known AI incident.""", json_schema_extra = { "linkml_meta": {'alias': 'hasVariant', 'domain': 'RiskIncident', 'domain_of': ['RiskIncident']} })
    author: Optional[str] = Field(default=None, description="""The author or authors of the incident report""", json_schema_extra = { "linkml_meta": {'alias': 'author', 'domain_of': ['Documentation', 'RiskIncident']} })
    source_uri: Optional[str] = Field(default=None, description="""The uri of the incident""", json_schema_extra = { "linkml_meta": {'alias': 'source_uri', 'domain_of': ['RiskIncident']} })
    isDetectedBy: Optional[list[str]] = Field(default=None, description="""A relationship where a risk, risk source, consequence, or impact is detected by a risk control.""", json_schema_extra = { "linkml_meta": {'alias': 'isDetectedBy',
         'domain': 'RiskConcept',
         'domain_of': ['RiskConcept'],
         'inverse': 'detectsRiskConcept'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Impact(RiskConcept, Entity):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:Impact',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk',
         'mixins': ['RiskConcept']})

    isDetectedBy: Optional[list[str]] = Field(default=None, description="""A relationship where a risk, risk source, consequence, or impact is detected by a risk control.""", json_schema_extra = { "linkml_meta": {'alias': 'isDetectedBy',
         'domain': 'RiskConcept',
         'domain_of': ['RiskConcept'],
         'inverse': 'detectsRiskConcept'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class IncidentStatus(Entity):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:IncidentStatus',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class IncidentConcludedclass(IncidentStatus):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:IncidentConcludedclass',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class IncidentHaltedclass(IncidentStatus):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:IncidentHaltedclass',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class IncidentMitigatedclass(IncidentStatus):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:IncidentMitigatedclass',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class IncidentNearMissclass(IncidentStatus):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:IncidentNearMissclass',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class IncidentOngoingclass(IncidentStatus):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:IncidentOngoingclass',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Severity(Entity):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:Severity',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Likelihood(Entity):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:Likelihood',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Consequence(Entity):
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'dpv:Consequence',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_risk'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    hasModelCard: Optional[list[str]] = Field(default=None, description="""A relationship to model card references.""", json_schema_extra = { "linkml_meta": {'alias': 'hasModelCard', 'domain_of': ['BaseAi']} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    performsTask: Optional[list[str]] = Field(default=None, description="""relationship indicating the AI tasks an AI model can perform.""", json_schema_extra = { "linkml_meta": {'alias': 'performsTask', 'domain_of': ['BaseAi']} })
    isProvidedBy: Optional[str] = Field(default=None, description="""A relationship indicating the AI model has been provided by an AI model provider.""", json_schema_extra = { "linkml_meta": {'alias': 'isProvidedBy',
         'domain_of': ['BaseAi'],
         'slot_uri': 'airo:isProvidedBy'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    hasModelCard: Optional[list[str]] = Field(default=None, description="""A relationship to model card references.""", json_schema_extra = { "linkml_meta": {'alias': 'hasModelCard', 'domain_of': ['BaseAi']} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    performsTask: Optional[list[str]] = Field(default=None, description="""relationship indicating the AI tasks an AI model can perform.""", json_schema_extra = { "linkml_meta": {'alias': 'performsTask', 'domain_of': ['BaseAi']} })
    isProvidedBy: Optional[str] = Field(default=None, description="""A relationship indicating the AI model has been provided by an AI model provider.""", json_schema_extra = { "linkml_meta": {'alias': 'isProvidedBy',
         'domain_of': ['BaseAi'],
         'slot_uri': 'airo:isProvidedBy'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    hasModelCard: Optional[list[str]] = Field(default=None, description="""A relationship to model card references.""", json_schema_extra = { "linkml_meta": {'alias': 'hasModelCard', 'domain_of': ['BaseAi']} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    performsTask: Optional[list[str]] = Field(default=None, description="""relationship indicating the AI tasks an AI model can perform.""", json_schema_extra = { "linkml_meta": {'alias': 'performsTask', 'domain_of': ['BaseAi']} })
    isProvidedBy: Optional[str] = Field(default=None, description="""A relationship indicating the AI agent has been provided by an AI systems provider.""", json_schema_extra = { "linkml_meta": {'alias': 'isProvidedBy',
         'domain_of': ['BaseAi'],
         'slot_uri': 'airo:isProvidedBy'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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

    hasEvaluation: Optional[list[str]] = Field(default=None, description="""A relationship indicating that an entity has an AI evaluation result.""", json_schema_extra = { "linkml_meta": {'alias': 'hasEvaluation',
         'domain_of': ['AiModel'],
         'slot_uri': 'dqv:hasQualityMeasurement'} })
    architecture: Optional[str] = Field(default=None, description="""A description of the architecture of an AI such as 'Decoder-only'.""", json_schema_extra = { "linkml_meta": {'alias': 'architecture', 'domain_of': ['AiModel']} })
    gpu_hours: Optional[int] = Field(default=None, description="""GPU consumption in terms of hours""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'gpu_hours', 'domain_of': ['AiModel']} })
    power_consumption_w: Optional[int] = Field(default=None, description="""power consumption in Watts""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'power_consumption_w', 'domain_of': ['AiModel']} })
    carbon_emitted: Optional[float] = Field(default=None, description="""The number of tons of carbon dioxide equivalent that are emitted during training""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'carbon_emitted',
         'domain_of': ['AiModel'],
         'unit': {'descriptive_name': 'tons of CO2 equivalent', 'symbol': 't CO2-eq'}} })
    hasRiskControl: Optional[list[str]] = Field(default=None, description="""Indicates the control measures associated with a system or component to modify risks.""", json_schema_extra = { "linkml_meta": {'alias': 'hasRiskControl',
         'domain_of': ['AiModel'],
         'slot_uri': 'airo:hasRiskControl'} })
    producer: Optional[str] = Field(default=None, description="""A relationship to the Organization instance which produces this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'producer', 'domain_of': ['BaseAi']} })
    hasModelCard: Optional[list[str]] = Field(default=None, description="""A relationship to model card references.""", json_schema_extra = { "linkml_meta": {'alias': 'hasModelCard', 'domain_of': ['BaseAi']} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    performsTask: Optional[list[str]] = Field(default=None, description="""relationship indicating the AI tasks an AI model can perform.""", json_schema_extra = { "linkml_meta": {'alias': 'performsTask', 'domain_of': ['BaseAi']} })
    isProvidedBy: Optional[str] = Field(default=None, description="""A relationship indicating the AI model has been provided by an AI model provider.""", json_schema_extra = { "linkml_meta": {'alias': 'isProvidedBy',
         'domain_of': ['BaseAi'],
         'slot_uri': 'airo:isProvidedBy'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    hasInputModality: Optional[list[str]] = Field(default=None, description="""A relationship indicating the input modalities supported by an AI component. Examples include text, image, video.""", json_schema_extra = { "linkml_meta": {'alias': 'hasInputModality', 'domain_of': ['LargeLanguageModel']} })
    hasOutputModality: Optional[list[str]] = Field(default=None, description="""A relationship indicating the output modalities supported by an AI component. Examples include text, image, video.""", json_schema_extra = { "linkml_meta": {'alias': 'hasOutputModality', 'domain_of': ['LargeLanguageModel']} })
    hasTrainingData: Optional[list[str]] = Field(default=None, description="""A relationship indicating the datasets an AI model was trained on.""", json_schema_extra = { "linkml_meta": {'alias': 'hasTrainingData',
         'domain_of': ['LargeLanguageModel'],
         'slot_uri': 'airo:hasTrainingData'} })
    fine_tuning: Optional[str] = Field(default=None, description="""A description of the fine-tuning mechanism(s) applied to a model.""", json_schema_extra = { "linkml_meta": {'alias': 'fine_tuning', 'domain_of': ['LargeLanguageModel']} })
    supported_languages: Optional[list[str]] = Field(default=None, description="""A list of languages, expressed as ISO two letter codes. For example, 'jp, fr, en, de'""", json_schema_extra = { "linkml_meta": {'alias': 'supported_languages', 'domain_of': ['LargeLanguageModel']} })
    isPartOf: Optional[str] = Field(default=None, description="""Annotation that a Large Language model is part of a family of models""", json_schema_extra = { "linkml_meta": {'alias': 'isPartOf',
         'domain_of': ['Risk', 'LargeLanguageModel', 'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    hasEvaluation: Optional[list[str]] = Field(default=None, description="""A relationship indicating that an entity has an AI evaluation result.""", json_schema_extra = { "linkml_meta": {'alias': 'hasEvaluation',
         'domain_of': ['AiModel'],
         'slot_uri': 'dqv:hasQualityMeasurement'} })
    architecture: Optional[str] = Field(default=None, description="""A description of the architecture of an AI such as 'Decoder-only'.""", json_schema_extra = { "linkml_meta": {'alias': 'architecture', 'domain_of': ['AiModel']} })
    gpu_hours: Optional[int] = Field(default=None, description="""GPU consumption in terms of hours""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'gpu_hours', 'domain_of': ['AiModel']} })
    power_consumption_w: Optional[int] = Field(default=None, description="""power consumption in Watts""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'power_consumption_w', 'domain_of': ['AiModel']} })
    carbon_emitted: Optional[float] = Field(default=None, description="""The number of tons of carbon dioxide equivalent that are emitted during training""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'carbon_emitted',
         'domain_of': ['AiModel'],
         'unit': {'descriptive_name': 'tons of CO2 equivalent', 'symbol': 't CO2-eq'}} })
    hasRiskControl: Optional[list[str]] = Field(default=None, description="""Indicates the control measures associated with a system or component to modify risks.""", json_schema_extra = { "linkml_meta": {'alias': 'hasRiskControl',
         'domain_of': ['AiModel'],
         'slot_uri': 'airo:hasRiskControl'} })
    producer: Optional[str] = Field(default=None, description="""A relationship to the Organization instance which produces this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'producer', 'domain_of': ['BaseAi']} })
    hasModelCard: Optional[list[str]] = Field(default=None, description="""A relationship to model card references.""", json_schema_extra = { "linkml_meta": {'alias': 'hasModelCard', 'domain_of': ['BaseAi']} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    performsTask: Optional[list[str]] = Field(default=None, description="""relationship indicating the AI tasks an AI model can perform.""", json_schema_extra = { "linkml_meta": {'alias': 'performsTask', 'domain_of': ['BaseAi']} })
    isProvidedBy: Optional[str] = Field(default=None, description="""A relationship indicating the AI model has been provided by an AI model provider.""", json_schema_extra = { "linkml_meta": {'alias': 'isProvidedBy',
         'domain_of': ['BaseAi'],
         'slot_uri': 'airo:isProvidedBy'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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

    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Input(Entity):
    """
    Input for which the system or component generates output.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'airo:Input',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_system'})

    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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

    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasDataset: Optional[list[str]] = Field(default=None, description="""A relationship to datasets that are used.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDataset', 'domain_of': ['AiEval']} })
    hasTasks: Optional[list[str]] = Field(default=None, description="""The tasks or evaluations the benchmark is intended to assess.""", json_schema_extra = { "linkml_meta": {'alias': 'hasTasks', 'domain_of': ['AiEval', 'BenchmarkMetadataCard']} })
    hasImplementation: Optional[list[str]] = Field(default=None, description="""A relationship to a implementation defining the risk evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'hasImplementation',
         'domain_of': ['AiEval'],
         'slot_uri': 'schema:url'} })
    hasUnitxtCard: Optional[list[str]] = Field(default=None, description="""A relationship to a Unitxt card defining the risk evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'hasUnitxtCard', 'domain_of': ['AiEval'], 'slot_uri': 'schema:url'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    hasRelatedRisk: Optional[list[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk',
         'any_of': [{'range': 'RiskConcept'}, {'range': 'Term'}],
         'domain': 'Any',
         'domain_of': ['Term',
                       'LLMQuestionPolicy',
                       'Action',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic']} })
    bestValue: Optional[str] = Field(default=None, description="""Annotation of the best possible result of the evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'bestValue', 'domain_of': ['AiEval']} })
    hasBenchmarkMetadata: Optional[list[str]] = Field(default=None, description="""A relationship to a Benchmark Metadata Card which contains metadata about the benchmark.""", json_schema_extra = { "linkml_meta": {'alias': 'hasBenchmarkMetadata',
         'domain': 'AiEval',
         'domain_of': ['AiEval'],
         'inverse': 'describesAiEval'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class BenchmarkMetadataCard(Entity):
    """
    Benchmark metadata cards offer a standardized way to document LLM benchmarks clearly and transparently. Inspired by Model Cards and Datasheets, Benchmark metadata cards help researchers and practitioners understand exactly what benchmarks test, how they relate to real-world risks, and how to interpret their results responsibly.  This is an implementation of the design set out in 'BenchmarkCards: Large Language Model and Risk Reporting' (https://doi.org/10.48550/arXiv.2410.12974)
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'nexus:benchmarkmetadatacard',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_eval'})

    describesAiEval: Optional[list[str]] = Field(default=None, description="""A relationship where a BenchmarkMetadataCard describes and AI evaluation (benchmark).""", json_schema_extra = { "linkml_meta": {'alias': 'describesAiEval',
         'domain': 'BenchmarkMetadataCard',
         'domain_of': ['BenchmarkMetadataCard'],
         'inverse': 'hasBenchmarkMetadata'} })
    hasDataType: Optional[list[str]] = Field(default=None, description="""The type of data used in the benchmark (e.g., text, images, or multi-modal)""", json_schema_extra = { "linkml_meta": {'alias': 'hasDataType', 'domain_of': ['BenchmarkMetadataCard']} })
    hasDomains: Optional[list[str]] = Field(default=None, description="""The specific domains or areas where the benchmark is applied (e.g., natural language processing,computer vision).""", json_schema_extra = { "linkml_meta": {'alias': 'hasDomains', 'domain_of': ['BenchmarkMetadataCard']} })
    hasLanguages: Optional[list[str]] = Field(default=None, description="""The languages included in the dataset used by the benchmark (e.g., English, multilingual).""", json_schema_extra = { "linkml_meta": {'alias': 'hasLanguages', 'domain_of': ['BenchmarkMetadataCard']} })
    hasSimilarBenchmarks: Optional[list[str]] = Field(default=None, description="""Benchmarks that are closely related in terms of goals or data type.""", json_schema_extra = { "linkml_meta": {'alias': 'hasSimilarBenchmarks', 'domain_of': ['BenchmarkMetadataCard']} })
    hasResources: Optional[list[str]] = Field(default=None, description="""Links to relevant resources, such as repositories or papers related to the benchmark.""", json_schema_extra = { "linkml_meta": {'alias': 'hasResources', 'domain_of': ['BenchmarkMetadataCard']} })
    hasGoal: Optional[str] = Field(default=None, description="""The specific goal or primary use case the benchmark is designed for.""", json_schema_extra = { "linkml_meta": {'alias': 'hasGoal', 'domain_of': ['BenchmarkMetadataCard']} })
    hasAudience: Optional[str] = Field(default=None, description="""The intended audience, such as researchers, developers, policymakers, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'hasAudience', 'domain_of': ['BenchmarkMetadataCard']} })
    hasTasks: Optional[list[str]] = Field(default=None, description="""The tasks or evaluations the benchmark is intended to assess.""", json_schema_extra = { "linkml_meta": {'alias': 'hasTasks', 'domain_of': ['AiEval', 'BenchmarkMetadataCard']} })
    hasLimitations: Optional[list[str]] = Field(default=None, description="""Limitations in evaluating or addressing risks, such as gaps in demographic coverage or specific domains.""", json_schema_extra = { "linkml_meta": {'alias': 'hasLimitations', 'domain_of': ['BenchmarkMetadataCard']} })
    hasOutOfScopeUses: Optional[list[str]] = Field(default=None, description="""Use cases where the benchmark is not designed to be applied and could give misleading results.""", json_schema_extra = { "linkml_meta": {'alias': 'hasOutOfScopeUses', 'domain_of': ['BenchmarkMetadataCard']} })
    hasDataSource: Optional[list[str]] = Field(default=None, description="""The origin or source of the data used in the benchmark (e.g., curated datasets, user submissions).""", json_schema_extra = { "linkml_meta": {'alias': 'hasDataSource', 'domain_of': ['BenchmarkMetadataCard']} })
    hasDataSize: Optional[str] = Field(default=None, description="""The size of the dataset, including the number of data points or examples.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDataSize', 'domain_of': ['BenchmarkMetadataCard']} })
    hasDataFormat: Optional[str] = Field(default=None, description="""The structure and modality of the data (e.g., sentence pairs, question-answer format, tabular data).""", json_schema_extra = { "linkml_meta": {'alias': 'hasDataFormat', 'domain_of': ['BenchmarkMetadataCard']} })
    hasAnnotation: Optional[str] = Field(default=None, description="""The process used to annotate or label the dataset, including who or what performed the annotations (e.g., human annotators, automated processes).""", json_schema_extra = { "linkml_meta": {'alias': 'hasAnnotation', 'domain_of': ['BenchmarkMetadataCard']} })
    hasMethods: Optional[list[str]] = Field(default=None, description="""The evaluation techniques applied within the benchmark.""", json_schema_extra = { "linkml_meta": {'alias': 'hasMethods', 'domain_of': ['BenchmarkMetadataCard']} })
    hasMetrics: Optional[list[str]] = Field(default=None, description="""The specific performance metrics used to assess models (e.g., accuracy, F1 score, precision, recall).""", json_schema_extra = { "linkml_meta": {'alias': 'hasMetrics', 'domain_of': ['BenchmarkMetadataCard']} })
    hasCalculation: Optional[list[str]] = Field(default=None, description="""The way metrics are computed based on model outputs and the benchmark data.""", json_schema_extra = { "linkml_meta": {'alias': 'hasCalculation', 'domain_of': ['BenchmarkMetadataCard']} })
    hasInterpretation: Optional[list[str]] = Field(default=None, description="""How users should interpret the scores or results from the metrics.""", json_schema_extra = { "linkml_meta": {'alias': 'hasInterpretation', 'domain_of': ['BenchmarkMetadataCard']} })
    hasBaselineResults: Optional[str] = Field(default=None, description="""The results of well-known or widely used models to give context to new performance scores.""", json_schema_extra = { "linkml_meta": {'alias': 'hasBaselineResults', 'domain_of': ['BenchmarkMetadataCard']} })
    hasValidation: Optional[list[str]] = Field(default=None, description="""Measures taken to ensure that the benchmark provides valid and reliable evaluations.""", json_schema_extra = { "linkml_meta": {'alias': 'hasValidation', 'domain_of': ['BenchmarkMetadataCard']} })
    hasRelatedRisk: Optional[list[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk',
         'any_of': [{'range': 'RiskConcept'}, {'range': 'Term'}],
         'domain': 'Any',
         'domain_of': ['Term',
                       'LLMQuestionPolicy',
                       'Action',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic']} })
    hasDemographicAnalysis: Optional[str] = Field(default=None, description="""How the benchmark evaluates performance across different demographic groups (e.g., gender, race).""", json_schema_extra = { "linkml_meta": {'alias': 'hasDemographicAnalysis', 'domain_of': ['BenchmarkMetadataCard']} })
    hasConsiderationPrivacyAndAnonymity: Optional[str] = Field(default=None, description="""How any personal or sensitive data is handled and whether any anonymization techniques are applied.""", json_schema_extra = { "linkml_meta": {'alias': 'hasConsiderationPrivacyAndAnonymity',
         'domain_of': ['BenchmarkMetadataCard']} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    hasConsiderationConsentProcedures: Optional[str] = Field(default=None, description="""Information on how consent was obtained (if applicable), especially for datasets involving personal data.""", json_schema_extra = { "linkml_meta": {'alias': 'hasConsiderationConsentProcedures',
         'domain_of': ['BenchmarkMetadataCard']} })
    hasConsiderationComplianceWithRegulations: Optional[str] = Field(default=None, description="""Compliance with relevant legal or ethical regulations (if applicable).""", json_schema_extra = { "linkml_meta": {'alias': 'hasConsiderationComplianceWithRegulations',
         'domain_of': ['BenchmarkMetadataCard']} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    name: Optional[str] = Field(default=None, description="""The official name of the benchmark.""", json_schema_extra = { "linkml_meta": {'alias': 'name', 'domain_of': ['Entity', 'BenchmarkMetadataCard']} })
    overview: Optional[str] = Field(default=None, description="""A brief description of the benchmark's main goals and scope.""", json_schema_extra = { "linkml_meta": {'alias': 'overview', 'domain_of': ['BenchmarkMetadataCard']} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
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
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasDataset: Optional[list[str]] = Field(default=None, description="""A relationship to datasets that are used.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDataset', 'domain_of': ['AiEval']} })
    hasTasks: Optional[list[str]] = Field(default=None, description="""The tasks or evaluations the benchmark is intended to assess.""", json_schema_extra = { "linkml_meta": {'alias': 'hasTasks', 'domain_of': ['AiEval', 'BenchmarkMetadataCard']} })
    hasImplementation: Optional[list[str]] = Field(default=None, description="""A relationship to a implementation defining the risk evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'hasImplementation',
         'domain_of': ['AiEval'],
         'slot_uri': 'schema:url'} })
    hasUnitxtCard: Optional[list[str]] = Field(default=None, description="""A relationship to a Unitxt card defining the risk evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'hasUnitxtCard', 'domain_of': ['AiEval'], 'slot_uri': 'schema:url'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    hasRelatedRisk: Optional[list[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk',
         'any_of': [{'range': 'RiskConcept'}, {'range': 'Term'}],
         'domain': 'Any',
         'domain_of': ['Term',
                       'LLMQuestionPolicy',
                       'Action',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic']} })
    bestValue: Optional[str] = Field(default=None, description="""Annotation of the best possible result of the evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'bestValue', 'domain_of': ['AiEval']} })
    hasBenchmarkMetadata: Optional[list[str]] = Field(default=None, description="""A relationship to a Benchmark Metadata Card which contains metadata about the benchmark.""", json_schema_extra = { "linkml_meta": {'alias': 'hasBenchmarkMetadata',
         'domain': 'AiEval',
         'domain_of': ['AiEval'],
         'inverse': 'describesAiEval'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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

    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasDataset: Optional[list[str]] = Field(default=None, description="""A relationship to datasets that are used.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDataset', 'domain_of': ['AiEval']} })
    hasTasks: Optional[list[str]] = Field(default=None, description="""The tasks or evaluations the benchmark is intended to assess.""", json_schema_extra = { "linkml_meta": {'alias': 'hasTasks', 'domain_of': ['AiEval', 'BenchmarkMetadataCard']} })
    hasImplementation: Optional[list[str]] = Field(default=None, description="""A relationship to a implementation defining the risk evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'hasImplementation',
         'domain_of': ['AiEval'],
         'slot_uri': 'schema:url'} })
    hasUnitxtCard: Optional[list[str]] = Field(default=None, description="""A relationship to a Unitxt card defining the risk evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'hasUnitxtCard', 'domain_of': ['AiEval'], 'slot_uri': 'schema:url'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    hasRelatedRisk: Optional[list[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk',
         'any_of': [{'range': 'RiskConcept'}, {'range': 'Term'}],
         'domain': 'Any',
         'domain_of': ['Term',
                       'LLMQuestionPolicy',
                       'Action',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic']} })
    bestValue: Optional[str] = Field(default=None, description="""Annotation of the best possible result of the evaluation""", json_schema_extra = { "linkml_meta": {'alias': 'bestValue', 'domain_of': ['AiEval']} })
    hasBenchmarkMetadata: Optional[list[str]] = Field(default=None, description="""A relationship to a Benchmark Metadata Card which contains metadata about the benchmark.""", json_schema_extra = { "linkml_meta": {'alias': 'hasBenchmarkMetadata',
         'domain': 'AiEval',
         'domain_of': ['AiEval'],
         'inverse': 'describesAiEval'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Adapter(LargeLanguageModel, Entity):
    """
    Adapter-based methods add extra trainable parameters after the attention and fully-connected layers of a frozen pretrained model to reduce memory-usage and speed up training. The adapters are typically small but demonstrate comparable performance to a fully finetuned model and enable training larger models with fewer resources. (https://huggingface.co/docs/peft/en/conceptual_guides/adapter)
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_intrinsic',
         'mixins': ['LargeLanguageModel']})

    hasAdapterType: Optional[AdapterType] = Field(default=None, description="""The Adapter type, for example: LORA, ALORA, X-LORA""", json_schema_extra = { "linkml_meta": {'alias': 'hasAdapterType', 'domain_of': ['Adapter']} })
    isDefinedByVocabulary: Optional[str] = Field(default=None, description="""A relationship where a term or a term group is defined by a vocabulary""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByVocabulary',
         'domain_of': ['Term', 'Adapter', 'LLMIntrinsic'],
         'slot_uri': 'schema:isPartOf'} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    hasLicense: Optional[str] = Field(default=None, description="""Indicates licenses associated with a resource""", json_schema_extra = { "linkml_meta": {'alias': 'hasLicense',
         'domain_of': ['Dataset',
                       'Documentation',
                       'Vocabulary',
                       'RiskTaxonomy',
                       'BaseAi',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter'],
         'slot_uri': 'airo:hasLicense'} })
    hasRelatedRisk: Optional[list[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk',
         'any_of': [{'range': 'RiskConcept'}, {'range': 'Term'}],
         'domain': 'Any',
         'domain_of': ['Term',
                       'LLMQuestionPolicy',
                       'Action',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic']} })
    adaptsModel: Optional[str] = Field(default=None, description="""The LargeLanguageModel being adapted""", json_schema_extra = { "linkml_meta": {'alias': 'adaptsModel', 'domain_of': ['Adapter']} })
    numParameters: Optional[int] = Field(default=None, description="""A property indicating the number of parameters in a LLM.""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'numParameters', 'domain_of': ['LargeLanguageModel']} })
    numTrainingTokens: Optional[int] = Field(default=None, description="""The number of tokens a AI model was trained on.""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'numTrainingTokens', 'domain_of': ['LargeLanguageModel']} })
    contextWindowSize: Optional[int] = Field(default=None, description="""The total length, in bytes, of an AI model's context window.""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'contextWindowSize', 'domain_of': ['LargeLanguageModel']} })
    hasInputModality: Optional[list[str]] = Field(default=None, description="""A relationship indicating the input modalities supported by an AI component. Examples include text, image, video.""", json_schema_extra = { "linkml_meta": {'alias': 'hasInputModality', 'domain_of': ['LargeLanguageModel']} })
    hasOutputModality: Optional[list[str]] = Field(default=None, description="""A relationship indicating the output modalities supported by an AI component. Examples include text, image, video.""", json_schema_extra = { "linkml_meta": {'alias': 'hasOutputModality', 'domain_of': ['LargeLanguageModel']} })
    hasTrainingData: Optional[list[str]] = Field(default=None, description="""A relationship indicating the datasets an AI model was trained on.""", json_schema_extra = { "linkml_meta": {'alias': 'hasTrainingData',
         'domain_of': ['LargeLanguageModel'],
         'slot_uri': 'airo:hasTrainingData'} })
    fine_tuning: Optional[str] = Field(default=None, description="""A description of the fine-tuning mechanism(s) applied to a model.""", json_schema_extra = { "linkml_meta": {'alias': 'fine_tuning', 'domain_of': ['LargeLanguageModel']} })
    supported_languages: Optional[list[str]] = Field(default=None, description="""A list of languages, expressed as ISO two letter codes. For example, 'jp, fr, en, de'""", json_schema_extra = { "linkml_meta": {'alias': 'supported_languages', 'domain_of': ['LargeLanguageModel']} })
    isPartOf: Optional[str] = Field(default=None, description="""Annotation that a Large Language model is part of a family of models""", json_schema_extra = { "linkml_meta": {'alias': 'isPartOf',
         'domain_of': ['Risk', 'LargeLanguageModel', 'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    hasEvaluation: Optional[list[str]] = Field(default=None, description="""A relationship indicating that an entity has an AI evaluation result.""", json_schema_extra = { "linkml_meta": {'alias': 'hasEvaluation',
         'domain_of': ['AiModel'],
         'slot_uri': 'dqv:hasQualityMeasurement'} })
    architecture: Optional[str] = Field(default=None, description="""A description of the architecture of an AI such as 'Decoder-only'.""", json_schema_extra = { "linkml_meta": {'alias': 'architecture', 'domain_of': ['AiModel']} })
    gpu_hours: Optional[int] = Field(default=None, description="""GPU consumption in terms of hours""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'gpu_hours', 'domain_of': ['AiModel']} })
    power_consumption_w: Optional[int] = Field(default=None, description="""power consumption in Watts""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'power_consumption_w', 'domain_of': ['AiModel']} })
    carbon_emitted: Optional[float] = Field(default=None, description="""The number of tons of carbon dioxide equivalent that are emitted during training""", ge=0, json_schema_extra = { "linkml_meta": {'alias': 'carbon_emitted',
         'domain_of': ['AiModel'],
         'unit': {'descriptive_name': 'tons of CO2 equivalent', 'symbol': 't CO2-eq'}} })
    hasRiskControl: Optional[list[str]] = Field(default=None, description="""Indicates the control measures associated with a system or component to modify risks.""", json_schema_extra = { "linkml_meta": {'alias': 'hasRiskControl',
         'domain_of': ['AiModel'],
         'slot_uri': 'airo:hasRiskControl'} })
    producer: Optional[str] = Field(default=None, description="""A relationship to the Organization instance which produces this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'producer', 'domain_of': ['BaseAi']} })
    hasModelCard: Optional[list[str]] = Field(default=None, description="""A relationship to model card references.""", json_schema_extra = { "linkml_meta": {'alias': 'hasModelCard', 'domain_of': ['BaseAi']} })
    performsTask: Optional[list[str]] = Field(default=None, description="""relationship indicating the AI tasks an AI model can perform.""", json_schema_extra = { "linkml_meta": {'alias': 'performsTask', 'domain_of': ['BaseAi']} })
    isProvidedBy: Optional[str] = Field(default=None, description="""A relationship indicating the AI model has been provided by an AI model provider.""", json_schema_extra = { "linkml_meta": {'alias': 'isProvidedBy',
         'domain_of': ['BaseAi'],
         'slot_uri': 'airo:isProvidedBy'} })


class LLMIntrinsic(Entity):
    """
    A capability that can be invoked through a well-defined API that is reasonably stable and independent of how the LLM intrinsic itself is implemented.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'class_uri': 'ai:Capability',
         'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_intrinsic'})

    hasRelatedRisk: Optional[list[str]] = Field(default=None, description="""A relationship where an entity relates to a risk""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedRisk',
         'any_of': [{'range': 'RiskConcept'}, {'range': 'Term'}],
         'domain': 'Any',
         'domain_of': ['Term',
                       'LLMQuestionPolicy',
                       'Action',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic']} })
    hasRelatedTerm: Optional[list[str]] = Field(default=None, description="""A relationship where an entity relates to a term""", json_schema_extra = { "linkml_meta": {'alias': 'hasRelatedTerm',
         'any_of': [{'range': 'RiskConcept'}, {'range': 'Term'}],
         'domain': 'Any',
         'domain_of': ['LLMIntrinsic']} })
    hasDocumentation: Optional[list[str]] = Field(default=None, description="""Indicates documentation associated with an entity.""", json_schema_extra = { "linkml_meta": {'alias': 'hasDocumentation',
         'domain_of': ['Dataset',
                       'Vocabulary',
                       'Term',
                       'Principle',
                       'RiskTaxonomy',
                       'Action',
                       'BaseAi',
                       'LargeLanguageModelFamily',
                       'AiEval',
                       'BenchmarkMetadataCard',
                       'Adapter',
                       'LLMIntrinsic'],
         'slot_uri': 'airo:hasDocumentation'} })
    isDefinedByVocabulary: Optional[str] = Field(default=None, description="""A relationship where a term or a term group is defined by a vocabulary""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByVocabulary',
         'domain_of': ['Term', 'Adapter', 'LLMIntrinsic'],
         'slot_uri': 'schema:isPartOf'} })
    hasAdapter: Optional[list[str]] = Field(default=None, description="""The Adapter for the intrinsic""", json_schema_extra = { "linkml_meta": {'alias': 'hasAdapter', 'domain': 'LLMIntrinsic', 'domain_of': ['LLMIntrinsic']} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class StakeholderGroup(Entity):
    """
    An AI system stakeholder grouping.
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_csiro_rai'})

    isDefinedByTaxonomy: Optional[str] = Field(default=None, description="""A relationship where a risk or a risk group is defined by a risk taxonomy""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByTaxonomy',
         'domain_of': ['Policy',
                       'RiskGroup',
                       'Risk',
                       'RiskControl',
                       'Action',
                       'RiskIncident',
                       'StakeholderGroup',
                       'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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


class Stakeholder(Entity):
    """
    An AI system stakeholder for Responsible AI governance (e.g., AI governors, users, consumers).
    """
    linkml_meta: ClassVar[LinkMLMeta] = LinkMLMeta({'from_schema': 'https://ibm.github.io/risk-atlas-nexus/ontology/ai_csiro_rai',
         'slot_usage': {'isPartOf': {'description': 'A relationship where a '
                                                    'stakeholder is part of a '
                                                    'stakeholder group',
                                     'name': 'isPartOf',
                                     'range': 'StakeholderGroup'}}})

    isDefinedByTaxonomy: Optional[str] = Field(default=None, description="""A relationship where a risk or a risk group is defined by a risk taxonomy""", json_schema_extra = { "linkml_meta": {'alias': 'isDefinedByTaxonomy',
         'domain_of': ['Policy',
                       'RiskGroup',
                       'Risk',
                       'RiskControl',
                       'Action',
                       'RiskIncident',
                       'StakeholderGroup',
                       'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    isPartOf: Optional[str] = Field(default=None, description="""A relationship where a stakeholder is part of a stakeholder group""", json_schema_extra = { "linkml_meta": {'alias': 'isPartOf',
         'domain_of': ['Risk', 'LargeLanguageModel', 'Stakeholder'],
         'slot_uri': 'schema:isPartOf'} })
    id: str = Field(default=..., description="""A unique identifier to this instance of the model element. Example identifiers include UUID, URI, URN, etc.""", json_schema_extra = { "linkml_meta": {'alias': 'id', 'domain_of': ['Entity'], 'slot_uri': 'schema:identifier'} })
    name: Optional[str] = Field(default=None, description="""A text name of this instance.""", json_schema_extra = { "linkml_meta": {'alias': 'name',
         'domain_of': ['Entity', 'BenchmarkMetadataCard'],
         'slot_uri': 'schema:name'} })
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

    organizations: Optional[list[Organization]] = Field(default=None, description="""A list of organizations""", json_schema_extra = { "linkml_meta": {'alias': 'organizations', 'domain_of': ['Container']} })
    licenses: Optional[list[License]] = Field(default=None, description="""A list of licenses""", json_schema_extra = { "linkml_meta": {'alias': 'licenses', 'domain_of': ['Container']} })
    modalities: Optional[list[Modality]] = Field(default=None, description="""A list of AI modalities""", json_schema_extra = { "linkml_meta": {'alias': 'modalities', 'domain_of': ['Container']} })
    aitasks: Optional[list[AiTask]] = Field(default=None, description="""A list of AI tasks""", json_schema_extra = { "linkml_meta": {'alias': 'aitasks', 'domain_of': ['Container']} })
    documents: Optional[list[Documentation]] = Field(default=None, description="""A list of documents""", json_schema_extra = { "linkml_meta": {'alias': 'documents', 'domain_of': ['Container']} })
    datasets: Optional[list[Dataset]] = Field(default=None, description="""A list of data sets""", json_schema_extra = { "linkml_meta": {'alias': 'datasets', 'domain_of': ['Container']} })
    llmintrinsics: Optional[list[LLMIntrinsic]] = Field(default=None, description="""A list of LLMIntrinsics""", json_schema_extra = { "linkml_meta": {'alias': 'llmintrinsics', 'domain_of': ['Container']} })
    adapters: Optional[list[Adapter]] = Field(default=None, description="""A list of Adapters""", json_schema_extra = { "linkml_meta": {'alias': 'adapters', 'domain_of': ['Container']} })
    taxonomies: Optional[list[RiskTaxonomy]] = Field(default=None, description="""A list of AI risk taxonomies""", json_schema_extra = { "linkml_meta": {'alias': 'taxonomies', 'domain_of': ['Container']} })
    vocabularies: Optional[list[Vocabulary]] = Field(default=None, description="""A list of vocabularies""", json_schema_extra = { "linkml_meta": {'alias': 'vocabularies', 'domain_of': ['Container']} })
    riskgroups: Optional[list[RiskGroup]] = Field(default=None, description="""A list of AI risk groups""", json_schema_extra = { "linkml_meta": {'alias': 'riskgroups', 'domain_of': ['Container']} })
    risks: Optional[list[Risk]] = Field(default=None, description="""A list of AI risks""", json_schema_extra = { "linkml_meta": {'alias': 'risks', 'domain_of': ['Container']} })
    riskcontrols: Optional[list[RiskControl]] = Field(default=None, description="""A list of AI risk controls""", json_schema_extra = { "linkml_meta": {'alias': 'riskcontrols', 'domain_of': ['Container']} })
    riskincidents: Optional[list[RiskIncident]] = Field(default=None, description="""A list of AI risk incidents""", json_schema_extra = { "linkml_meta": {'alias': 'riskincidents', 'domain_of': ['Container']} })
    terms: Optional[list[Term]] = Field(default=None, description="""A list of terms from a vocabulary""", json_schema_extra = { "linkml_meta": {'alias': 'terms', 'domain_of': ['Container']} })
    stakeholdergroups: Optional[list[StakeholderGroup]] = Field(default=None, description="""A list of AI stakeholder groups""", json_schema_extra = { "linkml_meta": {'alias': 'stakeholdergroups', 'domain_of': ['Container']} })
    stakeholders: Optional[list[Stakeholder]] = Field(default=None, description="""A list of stakeholders""", json_schema_extra = { "linkml_meta": {'alias': 'stakeholders', 'domain_of': ['Container']} })
    actions: Optional[list[Action]] = Field(default=None, description="""A list of risk related actions""", json_schema_extra = { "linkml_meta": {'alias': 'actions', 'domain_of': ['Container']} })
    evaluations: Optional[list[AiEval]] = Field(default=None, description="""A list of AI evaluation methods""", json_schema_extra = { "linkml_meta": {'alias': 'evaluations', 'domain_of': ['Container']} })
    aievalresults: Optional[list[AiEvalResult]] = Field(default=None, description="""A list of AI evaluation results""", json_schema_extra = { "linkml_meta": {'alias': 'aievalresults', 'domain_of': ['Container']} })
    benchmarkmetadatacards: Optional[list[BenchmarkMetadataCard]] = Field(default=None, description="""A list of AI evaluation benchmark metadata cards""", json_schema_extra = { "linkml_meta": {'alias': 'benchmarkmetadatacards', 'domain_of': ['Container']} })
    aimodelfamilies: Optional[list[LargeLanguageModelFamily]] = Field(default=None, description="""A list of AI model families""", json_schema_extra = { "linkml_meta": {'alias': 'aimodelfamilies', 'domain_of': ['Container']} })
    aimodels: Optional[list[LargeLanguageModel]] = Field(default=None, description="""A list of AI models""", json_schema_extra = { "linkml_meta": {'alias': 'aimodels', 'domain_of': ['Container']} })
    policies: Optional[list[Policy]] = Field(default=None, description="""A list of policies""", json_schema_extra = { "linkml_meta": {'alias': 'policies', 'domain_of': ['Container']} })
    llmquestionpolicies: Optional[list[LLMQuestionPolicy]] = Field(default=None, description="""A list of LLM question policies""", json_schema_extra = { "linkml_meta": {'alias': 'llmquestionpolicies', 'domain_of': ['Container']} })
    rules: Optional[list[Rule]] = Field(default=None, description="""A list of rules""", json_schema_extra = { "linkml_meta": {'alias': 'rules', 'domain_of': ['Container']} })
    prohibitions: Optional[list[Prohibition]] = Field(default=None, description="""A list of prohibitions""", json_schema_extra = { "linkml_meta": {'alias': 'prohibitions', 'domain_of': ['Container']} })
    permissions: Optional[list[Permission]] = Field(default=None, description="""A list of Permissions""", json_schema_extra = { "linkml_meta": {'alias': 'permissions', 'domain_of': ['Container']} })
    obligations: Optional[list[Obligation]] = Field(default=None, description="""A list of Obligations""", json_schema_extra = { "linkml_meta": {'alias': 'obligations', 'domain_of': ['Container']} })
    principles: Optional[list[Principle]] = Field(default=None, description="""A list of Principles""", json_schema_extra = { "linkml_meta": {'alias': 'principles', 'domain_of': ['Container']} })


# Model rebuild
# see https://pydantic-docs.helpmanual.io/usage/models/#rebuilding-a-model
Entity.model_rebuild()
Organization.model_rebuild()
License.model_rebuild()
Dataset.model_rebuild()
Documentation.model_rebuild()
Fact.model_rebuild()
Vocabulary.model_rebuild()
Term.model_rebuild()
Principle.model_rebuild()
Policy.model_rebuild()
LLMQuestionPolicy.model_rebuild()
Rule.model_rebuild()
Permission.model_rebuild()
Prohibition.model_rebuild()
Obligation.model_rebuild()
RiskTaxonomy.model_rebuild()
RiskConcept.model_rebuild()
RiskGroup.model_rebuild()
Risk.model_rebuild()
RiskControl.model_rebuild()
Action.model_rebuild()
RiskIncident.model_rebuild()
Impact.model_rebuild()
IncidentStatus.model_rebuild()
IncidentConcludedclass.model_rebuild()
IncidentHaltedclass.model_rebuild()
IncidentMitigatedclass.model_rebuild()
IncidentNearMissclass.model_rebuild()
IncidentOngoingclass.model_rebuild()
Severity.model_rebuild()
Likelihood.model_rebuild()
Consequence.model_rebuild()
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
Input.model_rebuild()
AiEval.model_rebuild()
AiEvalResult.model_rebuild()
BenchmarkMetadataCard.model_rebuild()
Question.model_rebuild()
Questionnaire.model_rebuild()
Adapter.model_rebuild()
LLMIntrinsic.model_rebuild()
AiOffice.model_rebuild()
StakeholderGroup.model_rebuild()
Stakeholder.model_rebuild()
Container.model_rebuild()
