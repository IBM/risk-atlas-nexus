# Add your own Chain of Thought (CoT) templates

CoT templates are designed to enhance interpretability, support systematic reasoning, and provide examples that aid model predictions, resulting in improved accuracy.

Risk Atlas Nexus currently supports adding custom CoT templates for [Auto-assist Risk Questionnaire](#auto-assist-risk-questionnaire) and [Risk Identification](#risk-identification).

## Auto-assist Risk Questionnaire

The auto-assist feature utilizes few-shot examples defined in the file `risk_atlas_nexus/data/templates/risk_questionnaire_cot.json` to predict the output of the risk questionnaire. The example of usage can be found in the notebook [autoassist_questionnaire.ipynb](../examples/notebooks/autoassist_questionnaire.ipynb)

### Customization

To adapt this auto-assist functionality to custom risk questionnaire, users need to provide their own set of questions, example intents, and corresponding answers in a json file such as in [risk_questionnaire_cot.json](https://github.com/IBM/risk-atlas-nexus/blob/main/src/risk_atlas_nexus/data/templates/risk_questionnaire_cot.json). This will enable the LLM to learn from these few-shot examples and generate responses for unseen queries.

There are two different template formats: the Zero-Shot method and the Few-Shot method.

### CoT Template - Zero Shot method

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

### CoT Template - Few Shot method

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

## Risk Identification

The risk identification feature utilizes few-shot examples defined in the file `risk_atlas_nexus/data/templates/risk_generation_cot.json` to predict potential risks associated with a usecase. The usage example is shown in the notebook [risk_identification.ipynb](../examples/notebooks/risk_identification.ipynb)

### Customization

To use custom risk CoT examples for a new taxonomy or an existing taxonomy, users must provide their own set of example usecase and associated risks in a JSON file such as in [risk_generation_cot.json](https://github.com/IBM/risk-atlas-nexus/blob/main/src/risk_atlas_nexus/data/templates/risk_generation_cot.json). This will enable the LLM to learn from these few-shot examples and generate better responses.

### CoT Template

Each taxonomy is associated with a list of examples, each containing usecase and associated risks.

```shell
  {
    "ibm-risk-atlas": [
        {
            "Usecase": "In a medical chatbot, generative AI can be employed to create a triage system that assesses patients' symptoms and provides immediate, contextually relevant advice based on their medical history and current condition. The chatbot can analyze the patient's input, identify potential medical issues, and offer tailored recommendations or insights to the patient or healthcare provider. This can help streamline the triage process, ensuring that patients receive the appropriate level of care and attention, and ultimately improving patient outcomes.",
            "Risks": [
                "Improper usage",
                "Incomplete advice",
                "Lack of model transparency",
                "Lack of system transparency",
                "Lack of training data transparency",
                "Data bias",
                "Uncertain data provenance",
                "Lack of data transparency",
                "Impact on human agency",
                "Impact on affected communities",
                "Improper retraining",
                "Inaccessible training data"
            ]
        }
        ...
    ]
    ...
  }
```

## Contribute the templates

If you would like to contribute your own templates to the project for others to use:

- Save your new template in the `risk_atlas_nexus/data/templates/` folder, push your changes and make a PR to the main project.
