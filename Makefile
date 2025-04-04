# The purpose of this makefile is to update pydantic classes and graph output

VERSION = $(shell git tag | tail -1)

MAKEFLAGS += --warn-undefined-variables

SCHEMA_NAME = ai-risk-ontology
LINKML_SCHEMA_NAME = ai-risk-ontology
SOURCE_SCHEMA_PATH = src/risk_atlas_nexus/ai_risk_ontology/schema
KG_DATA_PATH = src/risk_atlas_nexus/data/knowledge_graph
DATAMODEL_PATH = src/risk_atlas_nexus/ai_risk_ontology/datamodel

SHELL := bash
.SHELLFLAGS := -eu -o pipefail -c
.DEFAULT_GOAL := help
.DELETE_ON_ERROR:
.SUFFIXES:
.SECONDARY:

RUN=pipenv run

help: status
	@echo ""
	@echo "make test -- runs tests"
	@echo "make help -- show this help"
	@echo "make lift_mappings_from_tsv -- lift mappings from all tsv files to yaml directory."
	@echo "make compile_pydantic_model -- update pydantic classes"
	@echo "make regenerate_documentation -- regenerate the documentation"
	@echo "make regenerate_graph_output -- export the graph with all instances"
	@echo "make regenerate_owl_schema -- export the schema as OWL"
	@echo "make regenerate_risk_atlas_as_tex -- export the IBM AI risk atlas as .tex"
	@echo ""

status: 
	@echo "Project: $(SCHEMA_NAME)"
	@echo "Datafolder: $(SOURCE_SCHEMA_PATH)"

regenerate_documentation:
	gen-doc -d docs/ontology $(SOURCE_SCHEMA_PATH)/${LINKML_SCHEMA_NAME}.yaml

lift_mappings_from_tsv: 
	python ./src/risk_atlas_nexus/ai_risk_ontology/util/lifting/import_risk_mappings.py

compile_pydantic_model:
	gen-pydantic $(SOURCE_SCHEMA_PATH)/${LINKML_SCHEMA_NAME}.yaml > ${DATAMODEL_PATH}/ai_risk_ontology.py

regenerate_graph_output:
	python ./src/risk_atlas_nexus/ai_risk_ontology/util/export_graph.py

regenerate_owl_schema:
	gen-owl $(SOURCE_SCHEMA_PATH)/${LINKML_SCHEMA_NAME}.yaml --metadata-profile 'linkml' > graph_export/owl/${LINKML_SCHEMA_NAME}_schema.ttl

regenerate_risk_atlas_as_tex:
	python ./src/risk_atlas_nexus/ai_risk_ontology/util/export_risk_atlas_tex.py

lint_schema:
	linkml-lint $(SOURCE_SCHEMA_PATH)/${LINKML_SCHEMA_NAME}.yaml 
	
test:
	pytest
