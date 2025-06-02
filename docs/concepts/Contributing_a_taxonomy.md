# Contributing a taxonomy or CoT templates

This document will describe how to contribute custom  _taxonomy files_ and _CoT templates_ to the Risk Atlas Nexus
project.

1. [Overview](#overview)
    1. [What are the taxonomy files?](#what-are-the-taxonomy-files)
    2. [How are mappings created between taxonomies?](#how-are-mappings-maintained-between-taxonomies)
2. [Add your own taxonomies/risks/actions](#add-your-own-taxonomiesrisksactions)
    1. [An example addition of a risk](#an-example-addition-of-a-risk)
3. [Add your own mappings](#add-your-own-mappings)
4. [Add your own Chain of Thought templates](#add-your-own-chain-of-thought-templates)
    1. [Contribute the templates](#contribute-the-templates)

## Overview

You may wish to contribute your own taxonomy definitions, entity mappings or Chain of Thought templates to the Risk
Atlas Nexus project. We welcome any contributions.

Please familiarise yourself with the
general [contribution guidelines](https://github.com/ibm/risk-atlas-nexus/CONTRIBUTING.md) and set up your own fork of
the project.

### What are the taxonomy files?

The Risk Atlas Nexus ontology combines the AI risk view (taxonomies, risks, actions) with an AI model view (AI systems,
AI models, model evaluations) into one schema. These are loaded into the system with LinkML, through yaml
representations, stored in
the [knowledge graph data folder](https://github.com/ibm/risk-atlas-nexus/src/risk_atlas_nexus/data/knowledge_graph/).

### How are mappings maintained between taxonomies?

To express some semantically meaningful mapping between risks from different taxonomies, we have used the [Simple
Standard for Sharing Ontological Mappings (SSSOM)](https://academic.oup.com/database/article/doi/10.1093/database/baac035/6591806)
. The mappings are maintained in SSOM TSV files and are converted to LinkML data
YAML using Python helper scripts.

## Add your own taxonomies/risks/actions

You can add your own risk definitions by adding yaml
to [knowledge graph data folder](https://github.com/ibm/risk-atlas-nexus/src/risk_atlas_nexus/data/knowledge_graph/).
Ensure the entries comply with [the schema](../ontology/index.md)

### An example addition of a risk

If you would like to contribute your own taxonomy files to the project for others to use, add one or more yaml files to
the  [knowledge graph data directory](https://github.com/IBM/risk-atlas-nexus/tree/main/src/risk_atlas_nexus/data/knowledge_graph/)
.

For example, to add a new risk, create a file with the following sample content, and add it to the knowledge graph data
directory.

```
- id: my-own-risk
  name: A very risky AI behaviour
  description: An LLM-based system is often very risky
  isDefinedByTaxonomy: my-taxonomy
```

## Add your own mappings

1. Prepare a TSV file either
    1. manually, or
    2. semi-automatically, with aid of the python notebook [Prepare taxonomy mappings](../examples/notebooks/Prepare_taxonomy_mappings.ipynb).
2. Verify the TSV mappings are correct.
3. Lift all the TSV files to yaml mapping entries by executing `make lift_mappings_from_tsv`.
4. Ensure the changes were saved to the [knowledge graph data mapping folder](https://github.com/ibm/risk-atlas-nexus/src/risk_atlas_nexus/data/knowledge_graph/mapping/) .

### Save your changes
- Save your new taxonomy files in
  the [knowledge graph data directory](https://github.com/IBM/risk-atlas-nexus/tree/main/src/risk_atlas_nexus/data/knowledge_graph)
  . Push your changes and make a PR to the main project.

## Add your own Chain of Thought templates

The template used in `risk_atlas_nexus/data/templates/risk_questionnaire_cot.json` defines the few-shot examples which are use for the [auto-assist](../examples/notebooks/autoassist_questionnaire.ipynb) functionality.

**Customization:**

To adapt this auto-assist functionality to custom questionnaires, users need to provide their own set of questions, example intents, and corresponding answers in a json file (e.g., `risk_questionnaire_cot.json`). This will enable the LLM to learn from these few-shot examples and generate responses for unseen queries.

**Template Structure: Zero Shot**

Each question is accompanied by corresponding examples provided as an empty list.

```shell
  [
      {
          "question": "In which environment is the system used?",
          "cot_examples": []
      }
      ...
  ]
```

**Template Structure: Few Shot**

Each question is associated with a list of examples, each containing intent, answer, and optional explanation.

```shell
  [
      {
          "question": "In which environment is the system used?",
          "cot_examples": [
            {
              "intent": "Find patterns in healthcare insurance claims",
              "answer": "Insurance Claims Processing or Risk Management or Data Analytics",
              "explanation": "The system might be used by an insurance company's claims processing department to analyze and identify patterns in healthcare insurance claims."
            },
            {
                "intent": "optimize supply chain management in Investment banks",
                "answer": "Treasury Departments or Asset Management Divisions or Private Banking Units",
                "explanation": null
            },
            ...
          ]
      }
      ...
  ]
```

### Contribute the templates

If you would like to contribute your own templates to the project for others to use:

- Save your new template in the `risk_atlas_nexus/data/templates/` folder, push your changes and make a PR to the main
  project.
