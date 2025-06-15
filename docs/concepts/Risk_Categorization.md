# Risk Categorization

Assess and categorize the severity of risks associated with an AI system usecase, which includes the domain, purpose, capabilities, AI user, and AI subject.

## Prompt Templates

The service categorizes risk severity, as defined in the EU AI Act, using the following prompt templates located  in `src/risk_atlas_nexus/blocks/prompt_templates.py`

 - RISK_SEVERITY_INSTRUCTION
 - RISK_SEVERITY_TEMPLATE

Both the above templates have been extracted from the source code of [**ExploreGen**](https://github.com/sanja7s/ExploreGen/blob/main/code/RiskGen_AI_Design_EU_AI_Act_12_July_2024.ipynb), a novel LLM framework designed to generate realistic and varied uses of AI technology while classifying their risk levels based on EU AI Act regulations.

### References

Templates credit goes to the original authors.
```
Scepanovic, Sanja (2025). ExploreGen [Source code]. GitHub. https://github.com/sanja7s/ExploreGen/blob/main/code/RiskGen_AI_Design_EU_AI_Act_12_July_2024.ipynb
```

```
Herdel, Viviane, Sanja Šćepanović, Edyta Bogucka, and Daniele Quercia. "ExploreGen:
Large language models for envisioning the uses and risks of AI technologies." In Proceedings
of the AAAI/ACM Conference on AI, Ethics, and Society, vol. 7, pp. 584-596. 2024.
```

## Service API

The API receives a list of usecases along with an inference engine instance for LLM evaluation. It extracts key attributes - domain, purpose, capabilities, AI user, and AI subject—from each usecase. These attributes are then sent to the categorization service `RiskSeverityCategorizer.categorize()` to determine the Risk Severity and retrieve related information.


**API:** RiskAtlasNexus.categorize_risk_severity()

**Params:**
 - usecases (List[str]):
        A List of strings describing AI usecases
 - inference_engine (InferenceEngine):
        An LLM inference engine

**Returns:**
 - results (List[Dict]):
        Results detailing risk categorization by usecase.


## Example Response

**Usecase**: A system is used by a consortium of universities and financial institutions to both assess student academic performance and determine their eligibility and risk level for student loans. The system automatically evaluates students' historical academic data, standardized test results, socio-economic background, behavioral data from educational platforms, and other digital footprints (e.g., attendance, participation, learning pace).

**Response:**
```
[
   {
      "Description":"The AI system intended to be used is a machine learning model that analyzes historical academic data, standardized test results, socio-economic background, behavioral data, and digital footprints to predict student academic performance and loan eligibility/risk level, utilizing natural language processing and predictive analytics.",
      "Classification":"High Risk",
      "AIActText":"Article 6(2) - AI systems intended to be used for the purpose of assessing the appropriate level of education that an individual will receive or will be able to access, in the context of or within educational and vocational training institutions at all levels.",
      "Reasoning":"The system falls under the 'High Risk' category as per Article 6(2) of the EU AI Act, specifically under the 'Education and vocational training' section. It is intended to assess the level of education for individuals, which can materially influence their educational and professional course, potentially affecting their ability to secure a livelihood. This system may perpetuate historical patterns of discrimination and violate the right to education and training, as well as the right not to be discriminated against."
   },
   ...
]
```

The response object has four fields:
 - Description: The description of the AI System inferred from the usecase.
 - Classification: The risk severity classification label
    1) Excluded,
    2) Prohibited,
    3) High-Risk Exception,
    4) High Risk, and
    5) Limited or Low Risk.
 - AIActText: EU AI Act section that closely resembles the AI System including any amendments.
 - Reasoning: Explanation of the risk classification.

Refer the notebook example [risk_categorization.ipynb](../examples/notebooks/risk_categorization.ipynb)
