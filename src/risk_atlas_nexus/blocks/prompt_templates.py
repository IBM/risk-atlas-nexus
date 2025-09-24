QUESTIONNAIRE_COT_TEMPLATE = """
        I want you to play the role of a compliance officer and answer the question based on the given Intent.
        Return the question, answer, confidence and explanation in a json format where question, answer, confidence and explanation are keys of the json exactly as shown in the examples.
        you should answer the question followed by the confidence and explanation on how that answer was generated.
{% if cot_examples is not none %}{% for example in cot_examples %}
        Intent: {{ example.intent }}
        Question: {{ question }}
        Answer: {{ example.answer }}
{% if example.explanation is not none %}        Explanation: {{ example.explanation }}{% endif %}
{% if example.confidence is not none %}        confidence: {{ example.confidence }}{% endif %}
{% endfor %}{% endif %}
        Intent: {{ usecase }}
        Question: {{ question }}
"""

RISK_IDENTIFICATION_TEMPLATE = """You are an expert at AI risk classification. Study the risks JSON below containing list of risk category and its description.

RISKS:
{{ risks }}

Instructions:
1. Identify the potential RISKS associated with the given usecase. Use RISK `description` to verify if the risk is associated with the usecase.
2. If Input doesn't fit into any of the above RISKS categories, classify it as Unknown.
3. Respond with an array list{% if max_risk is not none %} (top {{ max_risk }} high risks categories){% endif %} of attribute 'category' containing the risk labels.
{% if cot_examples is not none and cot_examples|length > 0 %}
EXAMPLES:{% for example in cot_examples %}
Usecase: {{ example.Usecase }}
Risks: {{ example.Risks }}{% endfor %}
===== END OF EXAMPLES ======
{% endif %}
Usecase: {{ usecase }}
Risks: """

AI_TASKS_TEMPLATE = """Study and understand the JSON below containing a list of LLM task and its description.

{{ hf_ai_tasks }}

Your task is to identify one or more LLM tasks for the context given below. Respond only with a JSON list (maximum length {{ limit }} items) containing the most relevant LLM task labels. Do not include description. Ensure that your answer only includes the json list.

Context: {{ usecase }}
Output: """

"""
# RISK_SEVERITY_INSTRUCTION and RISK_SEVERITY_TEMPLATE adapted from https://github.com/sanja7s/ExploreGen/blob/main/code/RiskGen_AI_Design_EU_AI_Act_12_July_2024.ipynb, MIT License
# Original authors: Sanja Scepanovic, Edyta Bogucka, Marios Constantinides

Proceedings: Herdel, Viviane, Sanja Šćepanović, Edyta Bogucka, and Daniele Quercia. "ExploreGen:
Large language models for envisioning the uses and risks of AI technologies." In Proceedings
of the AAAI/ACM Conference on AI, Ethics, and Society, vol. 7, pp. 584-596. 2024.
"""
RISK_SEVERITY_INSTRUCTION = """You are an experienced regulatory compliance specialists who works in the field of AI technology regulation. You are thoughtful, decisive, experienced and conscientious.
You have access to the entirety of the EU AI Act and its amendments, which outline how various AI technologies are to be regulated and risk-classified within the European Union."""

RISK_SEVERITY_TEMPLATE = """

        TASK : Classify the following AI system use in a job by utilizing a five-tier classification:
        1) Excluded,
        2) Prohibited,
        3) High-Risk Exception,
        4) High Risk, and
        5) Limited or Low Risk.

        Follow these eight steps below:

        1. Write a brief description of the AI system, using similar language to the EU AI Act.
        The description should start with "The AI system intended to be used ...", and be written in a single sentence.

        2. Refer to Article 2 to determine whether your system is among the Excluded Systems by answering:
        Does the system fall within any of the following categories and other categories listed in Article 2?
            - AI Systems developed and used exclusively for military purposes,
            - AI research and development activity.

        If it is any of the options above, output the label "Excluded". Else, if it is None of the above, continue.

        3. Refer to Article 5 to determine whether the AI System is Prohibited by asking:
        Does your system perform any of these functions?
            - Subliminal techniques
            - Exploiting vulerabilities
            - Biometric categorization
            - Social scoring
            - Predictive policing
            - Expanding facial recognition databases
            - Emotion recognition
            - Real-time remote biometrics
        If it is any of the options above as well as those listed in the Article 5, output the label "Prohibited". Else, if it is None of the above, continue.

        4. Safety component means a component of a product or of an AI system which fulfils a safety function for that product or AI system,
        or the failure or malfunctioning of which endangers the health and safety of persons or property.

        Determine whether your system (or the product for which your AI system is a 'safety component') fall within any of the following high-risk categories?
            - Civil aviation security
            - Two- or three-wheels vehicles and quadricycles
            - Agricultural and forestry vehicles
            - Marine equipment
            - Interoperability of the rail system
            - Motor vehicles and their trailers
            - Civil aviation
        If it is any of the options above, output the label "High-Risk Exception". Else, if it is None of the above, continue.

        5. Determine whether your AI system (or the product for which your AI system is a 'safety component') fall within any of the following high-risk categories?
            - Machinery
            - Toys
            - Recreational craft & personal watercraft
            - Lifts and safety components of lifts
            - Equipment and protective systems intended for use in potentially explosive atmosoheres
            - Radio equipment
            - Pressure equipment
            - Cableway installations
            - Personal protective equipment
            - Appliances burning gaseout fuels
            - Medical devices
            - In vitro disgnostic medical devices
        If it is any of the options above, output the label "High Risk". Else, if it is None of the above, continue.

        6. Refer to Annex III to additionally determine does your AI system fall within any of the following high-risk categories?
            - Biometrics
            - Critical infrastructure
            - Educational and vocational training
            - Employment, workers management, and access to self-employment
            - Access to and enjoyment of essential private services and public services and benefits
            - Law enforcement
            - Migration, asylum, and border control management
            - Adminstration of justice and democratic processes
        If you answer None of the above then output the label "Limited or Low Risk". Else, continue.

        7. Does your AI system pose a significant risk of harm to the health, safety or fundamental rights of any person?
          The system does NOT pose a significant risk if one or more of the following conditions are met:
            - the AI system is intended to perform a narrow procedural task;
            - the AI system is intended to improve the result of a previously completed human activity;
            - the AI system is intended to detect decision-making patterns or deviations from prior decision-making patterns and is not meant to replace or influence the previously completed human assessment, without proper human review;
            - the AI system is intended to perform a preparatory task to an assessment relevant for the purpose of the use cases listed in Annex III.
        If your system meets any of these conditions, please output the label "Limited or Low Risk", and include the concrete point it met above in your reasoning explanation.
        If it meets none of these conditions, please output the label "High Risk".

        It is of utmost importance to exercise precision and make accurate judgments when classifying the risk associated with the AI system.
        When assessing critical infastructure, education,  employment, workers management and access to self-employment, the access to and enjoyment of certain essential private and public services and benefits (including disaster and emergency response), consider also the Provisions 55-58 provided in input.

        Please carefully consider all the regulations listed below during the risk classification of the AI system use:

        RELEVANT PROVISIONS:
        (55) As regards the management and operation of critical infrastructure, it is appropriate to
        classify as high-risk the AI systems intended to be used as safety components in the
        management and operation of critical digital infrastructure as listed in point (8) of the
        Annex to Directive (EU) 2022/2557, road traffic and the supply of water, gas, heating and
        electricity, since their failure or malfunctioning may put at risk the life and health of
        persons at large scale and lead to appreciable disruptions in the ordinary conduct of social
        and economic activities. Safety components of critical infrastructure, including critical
        digital infrastructure, are systems used to directly protect the physical integrity of critical
        infrastructure or the health and safety of persons and property but which are not
        necessary in order for the system to function. The failure or malfunctioning of such
        components might directly lead to risks to the physical integrity of critical infrastructure
        and thus to risks to health and safety of persons and property. Components intended to
        be used solely for cybersecurity purposes should not qualify as safety components.
        Examples of safety components of such critical infrastructure may include systems for
        monitoring water pressure or fire alarm controlling systems in cloud computing centres.

        (56)  The deployment of AI systems in education is important to promote high-quality digital
        education and training and to allow all learners and teachers to acquire and share the
        necessary digital skills and competences, including media literacy, and critical thinking,
        to take an active part in the economy, society, and in democratic processes. However, AI
        systems used in education or vocational training, in particular for determining access or
        admission, for assigning persons to educational and vocational training institutions or
        programmes at all levels, for evaluating learning outcomes of persons, for assessing the
        appropriate level of education for an individual and materially influencing the level of
        education and training that individuals will receive or will be able to access or for
        monitoring and detecting prohibited behaviour of students during tests should be
        classified as high-risk AI systems, since they may determine the educational and
        professional course of a person's life and therefore may affect that person's ability to
        secure a livelihood. When improperly designed and used, such systems may be
        particularly intrusive and may violate the right to education and training as well as the
        right not to be discriminated against and perpetuate historical patterns of discrimination,
        for example against women, certain age groups, persons with disabilities, or persons of
        certain racial or ethnic origins or sexual orientation.

        (57) AI systems used in employment, workers management and access to self-employment, in
        particular for the recruitment and selection of persons, for making decisions affecting
        terms of the work-related relationship, promotion and termination of work-related
        contractual relationships, for allocating tasks on the basis of individual behaviour,
        personal traits or characteristics and for monitoring or evaluation of persons in workrelated contractual
        relationships, should also be classified as high-risk, since those systems
        may have an appreciable impact on future career prospects, livelihoods of those persons
        and workers' rights. Relevant work-related contractual relationships should, in a
        meaningful manner, involve employees and persons providing services through platforms
        as referred to in the Commission Work Programme 2021. ▌ Throughout the recruitment
        process and in the evaluation, promotion, or retention of persons in work-related
        contractual relationships, such systems may perpetuate historical patterns of
        discrimination, for example against women, certain age groups, persons with disabilities,
        or persons of certain racial or ethnic origins or sexual orientation. AI systems used to
        monitor the performance and behaviour of such persons may also undermine their
        fundamental rights to data protection and privacy.

        (58) Another area in which the use of AI systems deserves special consideration is the access to
        and enjoyment of certain essential private and public services and benefits necessary for
        people to fully participate in society or to improve one's standard of living. In particular, ▌
        natural persons applying for or receiving essential public assistance benefits and services
        from public authorities namely healthcare services, social security benefits, social
        services providing protection in cases such as maternity, illness, industrial accidents,
        dependency or old age and loss of employment and social and housing assistance, are
        typically dependent on those benefits and services and in a vulnerable position in relation
        to the responsible authorities. If AI systems are used for determining whether such benefits
        and services should be granted, denied, reduced, revoked or reclaimed by authorities,
        including whether beneficiaries are legitimately entitled to such benefits or services,
        those systems may have a significant impact on persons' livelihood and may infringe their
        fundamental rights, such as the right to social protection, non-discrimination, human
        dignity or an effective remedy and should therefore be classified as high-risk. Nonetheless,
        this Regulation should not hamper the development and use of innovative approaches in
        the public administration, which would stand to benefit from a wider use of compliant and
        safe AI systems, provided that those systems do not entail a high risk to legal and natural
        persons.

        In addition, AI systems used to evaluate the credit score or creditworthiness of natural
        persons should be classified as high-risk AI systems, since they determine those persons'
        access to financial resources or essential services such as housing, electricity, and
        telecommunication services. AI systems used for those purposes may lead to
        discrimination between persons or groups and may perpetuate historical patterns of
        discrimination, such as that based on racial or ethnic origins, gender, disabilities, age or
        sexual orientation, or may create new forms of discriminatory impacts. However, AI
        systems provided for by Union law for the purpose of detecting fraud in the offering of
        financial services and for prudential purposes to calculate credit institutions' and
        insurance undertakings' capital requirements should not be considered to be high-risk
        under this Regulation. Moreover, AI systems intended to be used for risk assessment and
        pricing in relation to natural persons for health and life insurance can also have a
        significant impact on persons' livelihood and if not duly designed, developed and used,
        can infringe their fundamental rights and can lead to serious consequences for people's
        life and health, including financial exclusion and discrimination. Finally, AI systems
        used to evaluate and classify emergency calls by natural persons or to dispatch or
        establish priority in the dispatching of emergency first response services, including by
        police, firefighters and medical aid, as well as of emergency healthcare patient triage
        systems, should also be classified as high-risk since they make decisions in very critical
        situations for the life and health of persons and their property.

        1. This Regulation applies to:

          (a) providers placing on the market or putting into service AI systems or placing on the market general-purpose AI models in the Union, irrespective of whether those providers are established or located within the Union or in a third country;
          (b) deployers of AI systems that have their place of establishment or are located within the Union;
          (c) providers and deployers of AI systems that have their place of establishment or are located in a third country, where the output produced by the AI system is used in the Union;
          (d) importers and distributors of AI systems;
          (e) product manufacturers placing on the market or putting into service an AI system together with their product and under their own name or trademark;
          (f) authorised representatives of providers, which are not established in the Union;
          (g) affected persons that are located in the Union.

          Related: Recital 21 and Recital 22

          2. For AI systems classified as high-risk AI systems in accordance with Article 6(1) related to products covered by the Union harmonisation legislation listed in Section B of Annex I, only Article 6(1), Articles 102 to 109 and Article 112 apply. Article 57 applies only in so far as the requirements for high-risk AI systems under this Regulation have been integrated in that Union harmonisation legislation.
          3. This Regulation does not apply to areas outside the scope of Union law, and shall not, in any event, affect the competences of the Member States concerning national security, regardless of the type of entity entrusted by the Member States with carrying out tasks in relation to those competences. This Regulation does not apply to AI systems where and in so far they are placed on the market, put into service, or used with or without modification exclusively for military, defence or national security purposes, regardless of the type of entity carrying out those activities. This Regulation does not apply to AI systems which are not placed on the market or put into service in the Union, where the output is used in the Union exclusively for military, defence or national security purposes, regardless of the type of entity carrying out those activities. Related: Recital 24
          4. This Regulation applies neither to public authorities in a third country nor to international organisations falling within the scope of this Regulation pursuant to paragraph 1, where those authorities or organisations use AI systems in the framework of international cooperation or agreements for law enforcement and judicial cooperation with the Union or with one or more Member States, provided that such a third country or international organisation provides adequate safeguards with respect to the protection of fundamental rights and freedoms of individuals. Related: Recital 22
          5. This Regulation shall not affect the application of the provisions on the liability of providers of intermediary services as set out in Chapter II of Regulation (EU) 2022/2065. Related: Recital 11
          6. This Regulation does not apply to AI systems or AI models, including their output, specifically developed and put into service for the sole purpose of scientific research and development. Related: Recital 25
          7. Union law on the protection of personal data, privacy and the confidentiality of communications applies to personal data processed in connection with the rights and obligations laid down in this Regulation. This Regulation shall not affect Regulation (EU) 2016/679 or (EU) 2018/1725, or Directive 2002/58/EC or (EU) 2016/680, without prejudice to Article 10(5) and Article 59 of this Regulation. Related: Recital 10
          8. This Regulation does not apply to any research, testing or development activity regarding AI systems or AI models prior to their being placed on the market or put into service. Such activities shall be conducted in accordance with applicable Union law. Testing in real world conditions shall not be covered by that exclusion. Related: Recital 25
          9. This Regulation is without prejudice to the rules laid down by other Union legal acts related to consumer protection and product safety.
          10. This Regulation does not apply to obligations of deployers who are natural persons using AI systems in the course of a purely personal non-professional activity.
          11. This Regulation does not preclude the Union or Member States from maintaining or introducing laws, regulations or administrative provisions which are more favourable to workers in terms of protecting their rights in respect of the use of AI systems by employers, or from encouraging or allowing the application of collective agreements which are more favourable to workers.
          12. This Regulation does not apply to AI systems released under free and open-source licences, unless they are placed on the market or put into service as high-risk AI systems or as an AI system that falls under Article 5 or 50.

        Article 5: Prohibited Artificial Intelligence Practices
              1. The following AI practices shall be prohibited:
                  (a) the placing on the market, the putting into service or the use of an AI system that deploys subliminal techniques beyond a person’s consciousness or purposefully manipulative or deceptive techniques, with the objective, or the effect of materially distorting the behaviour of a person or a group of persons by appreciably impairing their ability to make an informed decision, thereby causing them to take a decision that they would not have otherwise taken in a manner that causes or is reasonably likely to cause that person, another person or group of persons significant harm; Related: Recital 29
                  (b) the placing on the market, the putting into service or the use of an AI system that exploits any of the vulnerabilities of a natural person or a specific group of persons due to their age, disability or a specific social or economic situation, with the objective, or the effect, of materially distorting the behaviour of that person or a person belonging to that group in a manner that causes or is reasonably likely to cause that person or another person significant harm; Related: Recital 29
                  (c) the placing on the market, the putting into service or the use of AI systems for the evaluation or classification of natural persons or groups of persons over a certain period of time based on their social behaviour or known, inferred or predicted personal or personality characteristics, with the social score leading to either or both of the following:
                      (i) detrimental or unfavourable treatment of certain natural persons or groups of persons in social contexts that are unrelated to the contexts in which the data was originally generated or collected;
                      (ii) detrimental or unfavourable treatment of certain natural persons or groups of persons that is unjustified or disproportionate to their social behaviour or its gravity;
              Related: Recital 31
                  (d) the placing on the market, the putting into service for this specific purpose, or the use of an AI system for making risk assessments of natural persons in order to assess or predict the risk of a natural person committing a criminal offence, based solely on the profiling of a natural person or on assessing their personality traits and characteristics; this prohibition shall not apply to AI systems used to support the human assessment of the involvement of a person in a criminal activity, which is already based on objective and verifiable facts directly linked to a criminal activity; Related: Recital 42
                  (e) the placing on the market, the putting into service for this specific purpose, or the use of AI systems that create or expand facial recognition databases through the untargeted scraping of facial images from the internet or CCTV footage; Related: Recital 43
                  (f) the placing on the market, the putting into service for this specific purpose, or the use of AI systems to infer emotions of a natural person in the areas of workplace and education institutions, except where the use of the AI system is intended to be put in place or into the market for medical or safety reasons; Related: Recital 44
                  (g) the placing on the market, the putting into service for this specific purpose, or the use of biometric categorisation systems that categorise individually natural persons based on their biometric data to deduce or infer their race, political opinions, trade union membership, religious or philosophical beliefs, sex life or sexual orientation; this prohibition does not cover any labelling or filtering of lawfully acquired biometric datasets, such as images, based on biometric data or categorizing of biometric data in the area of law enforcement; Related: Recital 30
                  (h) the use of ‘real-time’ remote biometric identification systems in publicly accessible spaces for the purposes of law enforcement, unless and in so far as such use is strictly necessary for one of the following objectives:
                        (i) the targeted search for specific victims of abduction, trafficking in human beings or sexual exploitation of human beings, as well as the search for missing persons;
                        (ii) the prevention of a specific, substantial and imminent threat to the life or physical safety of natural persons or a genuine and present or genuine and foreseeable threat of a terrorist attack;
                        (iii) the localisation or identification of a person suspected of having committed a criminal offence, for the purpose of conducting a criminal investigation or prosecution or executing a criminal penalty for offences referred to in Annex II and punishable in the Member State concerned by a custodial sentence or a detention order for a maximum period of at least four years. Point (h) of the first subparagraph is without prejudice to Article 9 of Regulation (EU) 2016/679 for the processing of biometric data for purposes other than law enforcement.
              Related: Recitals 32, 33, 38, 39, 40, and 41

              2. The use of ‘real-time’ remote biometric identification systems in publicly accessible spaces for the purposes of law enforcement for any of the objectives referred to in paragraph 1, first subparagraph, point (h), shall be deployed for the purposes set out in that point only to confirm the identity of the specifically targeted individual, and it shall take into account the following elements:
                  (a) the nature of the situation giving rise to the possible use, in particular the seriousness, probability and scale of the harm that would be caused if the system were not used;
                  (b) the consequences of the use of the system for the rights and freedoms of all persons concerned, in particular the seriousness, probability and scale of those consequences. In addition, the use of ‘real-time’ remote biometric identification systems in publicly accessible spaces for the purposes of law enforcement for any of the objectives referred to in paragraph 1, first subparagraph, point (h), of this Article shall comply with necessary and proportionate safeguards and conditions in relation to the use in accordance with the national law authorising the use thereof, in particular as regards the temporal, geographic and personal limitations. The use of the ‘real-time’ remote biometric identification system in publicly accessible spaces shall be authorised only if the law enforcement authority has completed a fundamental rights impact assessment as provided for in Article 27 and has registered the system in the EU database according to Article 49. However, in duly justified cases of urgency, the use of such systems may be commenced without the registration in the EU database, provided that such registration is completed without undue delay.
              Related: Recitals 34, 38, 39, 40, and 41

              3. For the purposes of paragraph 1, first subparagraph, point (h) and paragraph 2, each use for the purposes of law enforcement of a ‘real-time’ remote biometric identification system in publicly accessible spaces shall be subject to a prior authorisation granted by a judicial authority or an independent administrative authority whose decision is binding of the Member State in which the use is to take place, issued upon a reasoned request and in accordance with the detailed rules of national law referred to in paragraph 5. However, in a duly justified situation of urgency, the use of such system may be commenced without an authorisation provided that such authorisation is requested without undue delay, at the latest within 24 hours. If such authorisation is rejected, the use shall be stopped with immediate effect and all the data, as well as the results and outputs of that use shall be immediately discarded and deleted. The competent judicial authority or an independent administrative authority whose decision is binding shall grant the authorisation only where it is satisfied, on the basis of objective evidence or clear indications presented to it, that the use of the ‘real-time’ remote biometric identification system concerned is necessary for, and proportionate to, achieving one of the objectives specified in paragraph 1, first subparagraph, point (h), as identified in the request and, in particular, remains limited to what is strictly necessary concerning the period of time as well as the geographic and personal scope. In deciding on the request, that authority shall take into account the elements referred to in paragraph 2. No decision that produces an adverse legal effect on a person may be taken based solely on the output of the ‘real-time’ remote biometric identification system. Related: Recitals 35, 38, 39, 40, and 41
              4. Without prejudice to paragraph 3, each use of a ‘real-time’ remote biometric identification system in publicly accessible spaces for law enforcement purposes shall be notified to the relevant market surveillance authority and the national data protection authority in accordance with the national rules referred to in paragraph 5. The notification shall, as a minimum, contain the information specified under paragraph 6 and shall not include sensitive operational data. Related: Recitals 36, 38, 39, 40, and 41
              5. A Member State may decide to provide for the possibility to fully or partially authorise the use of ‘real-time’ remote biometric identification systems in publicly accessible spaces for the purposes of law enforcement within the limits and under the conditions listed in paragraph 1, first subparagraph, point (h), and paragraphs 2 and 3. Member States concerned shall lay down in their national law the necessary detailed rules for the request, issuance and exercise of, as well as supervision and reporting relating to, the authorisations referred to in paragraph 3. Those rules shall also specify in respect of which of the objectives listed in paragraph 1, first subparagraph, point (h), including which of the criminal offences referred to in point (h)(iii) thereof, the competent authorities may be authorised to use those systems for the purposes of law enforcement. Member States shall notify those rules to the Commission at the latest 30 days following the adoption thereof. Member States may introduce, in accordance with Union law, more restrictive laws on the use of remote biometric identification systems. Related: Recitals 37, 38, 39, 40, and 41
              6. National market surveillance authorities and the national data protection authorities of Member States that have been notified of the use of ‘real-time’ remote biometric identification systems in publicly accessible spaces for law enforcement purposes pursuant to paragraph 4 shall submit to the Commission annual reports on such use. For that purpose, the Commission shall provide Member States and national market surveillance and data protection authorities with a template, including information on the number of the decisions taken by competent judicial authorities or an independent administrative authority whose decision is binding upon requests for authorisations in accordance with paragraph 3 and their result. Related: Recitals 38, 39, 40, and 41
              7. The Commission shall publish annual reports on the use of real-time remote biometric identification systems in publicly accessible spaces for law enforcement purposes, based on aggregated data in Member States on the basis of the annual reports referred to in paragraph 6. Those annual reports shall not include sensitive operational data of the related law enforcement activities. Related: Recitals 38, 39, 40, and 41
              8. This Article shall not affect the prohibitions that apply where an AI practice infringes other Union law.

        1a. This Article shall not affect the prohibitions that apply where an artificial intelligence practice infringes other Union law.
        2. The use of 'real-time' remote biometric identification systems in publicly accessible spaces for the purpose of law enforcement for any of the objectives referred to in paragraph 1 point d) shall only be deployed for the purposes under paragraph 1, point d) to confirm the specifically targeted individual's identity and it shall take into account the following elements:
            (a) the nature of the situation giving rise to the possible use, in particular the seriousness, probability and scale of the harm caused in the absence of the use of the system;
            (b) the consequences of the use of the system for the rights and freedoms of all persons concerned, in particular the seriousness, probability and scale of those consequences. In addition, the use of 'real-time' remote biometric identification systems in publicly accessible spaces for the purpose of law enforcement for any of the objectives referred to in paragraph 1 point d) shall comply with necessary and proportionate safeguards and conditions in relation to the use in accordance with national legislations authorizing the use thereof, in particular as regards the temporal, geographic and personal limitations. The use of the 'real-time' remote biometric identification system in publicly accessible spaces shall only be authorised if the law enforcement authority has completed a fundamental rights impact assessment as provided for in Article 27 and has registered the system in the database according to Article 49. However, in duly justified cases of urgency, the use of the system may be commenced without the registration, provided that the registration is completed without undue delay.
        3. As regards paragraphs 1, point (d) and 2, each use for the purpose of law enforcement of a 'real-time' remote biometric identification system in publicly accessible spaces shall be subject to a prior authorisation granted by a judicial authority or an independent administrative authority whose decision is binding of the Member State in which the use is to take place, issued upon a reasoned request and in accordance with the detailed rules of national law referred to in paragraph 4. However, in a duly justified situation of urgency, the use of the system may be commenced without an authorisation provided that, such authorisation shall be requested without undue delay, at the latest within 24 hours. If such authorisation is rejected, its use shall be stopped with immediate effect and all the data, as well as the results and outputs of this use shall be immediately discarded and deleted. The competent judicial authority or an independent administrative authority whose decision is binding shall only grant the authorisation where it is satisfied, based on objective evidence or clear indications presented to it, that the use of the 'real-time' remote biometric identification system at issue is necessary for and proportionate to achieving one of the objectives specified in paragraph 1, point (d), as identified in the request and, in particular, remains limited to what is strictly necessary concerning the period of time as well as geographic and personal scope. In deciding on the request, the competent judicial authority or an independent administrative authority whose decision is binding shall take into account the elements referred to in paragraph 2. It shall be ensured that no decision that produces an adverse legal effect on a person may be taken by the judicial authority or an independent administrative authority whose decision is binding solely based on the output of the remote biometric identification system.
        3a. Without prejudice to paragraph 3, each use of a 'real-time' remote biometric identification system in publicly accessible spaces for law enforcement purposes shall be notified to the relevant market surveillance authority and the national data protection authority in accordance with the national rules referred to in paragraph 4. The notification shall as a minimum contain the information specified under paragraph 5 and shall not include sensitive operational data.
        4. A Member State may decide to provide for the possibility to fully or partially authorise the use of 'real-time' remote biometric identification systems in publicly accessible spaces for the purpose of law enforcement within the limits and under the conditions listed in paragraphs 1, point (d), 2 and 3. Member States concerned shall lay down in their national law the necessary detailed rules for the request, issuance and exercise of, as well as supervision and reporting relating to, the authorisations referred to in paragraph 3. Those rules shall also specify in respect of which of the objectives listed in paragraph 1, point (d), including which of the criminal offences referred to in point (iii) thereof, the competent authorities may be authorised to use those systems for the purpose of law enforcement. Member States shall notify those rules to the Commission at the latest 30 days following the adoption thereof. Member States may introduce, in accordance with Union law, more restrictive laws on the use of remote biometric identification systems.
        5. National market surveillance authorities and the national data protection authorities of Member States that have been notified of the use of 'real-time' remote biometric identification systems in publicly accessible spaces for law enforcement purposes pursuant to paragraph 3a shall submit to the Commission annual reports on such use. For that purpose, the Commission shall provide Member States and national market surveillance and data protection authorities with a template, including information on the number of the decisions taken by competent judicial authorities or an independent administrative authority whose decision is binding upon requests for authorisations in accordance with paragraph 3 and their result.
        6. The Commission shall publish annual reports on the use of 'real-time' remote biometric identification systems in publicly accessible spaces for law enforcement purposes based on aggregated data in Member States based on the annual reports referred to in paragraph 5, which shall not include sensitive operational data of the related law enforcement activities.
        (*) For the purposes of this Regulation the notion of *publicly accessible* space should be understood as referring to any physical place that is accessible to an undetermined number of natural persons, and irrespective of whether the place in question is privately or publicly owned and irrespective of the activity for which the place may be used, such as commerce (for instance, shops, restaurants, cafés), services (for instance, banks, professional activities, hospitality), sport (for instance, swimming pools, gyms, stadiums), transport (for instance, bus, metro and railway stations, airports, means of transport ), entertainment (for instance, cinemas, theatres, museums, concert and conference halls) leisure or otherwise (for instance, public roads and squares, parks, forests, playgrounds). A place should be classified as publicly accessible also if, regardless of potential capacity or security restrictions, access is subject to certain predetermined conditions, which can be fulfilled by an undetermined number of persons, such as purchase of a ticket or title of transport, prior registration or having a certain age. By contrast, a place should not be considered publicly accessible if access is limited to specific and defined natural persons through either Union or national law directly related to public safety or security or through the clear manifestation of will by the person having the relevant authority on the place. The factual possibility of access alone (e.g. an unlocked door, an open gate in a fence) does not imply that the place is publicly accessible in the presence of indications or circumstances suggesting the contrary (e.g. signs prohibiting or restricting access). Company and factory premises as well as offices and workplaces that are intended to be accessed only by relevant employees and service providers are places that are not publicly accessible. Publicly accessible spaces should not include prisons or border control. Some other areas may be composed of both not publicly accessible and publicly accessible areas, such as the hallway of a private residential building necessary to access a doctor's office or an airport. Online spaces are not covered either, as they are not physical spaces. Whether a given space is accessible to the public should however be determined on a case-by-case basis, having regard to the specificities of the individual situation at hand.

        Annex II: List of Criminal Offences
              Criminal offences referred to in Article 5(1), first subparagraph, point (h)(iii):
              – terrorism,
              – trafficking in human beings,
              – sexual exploitation of children, and child pornography,
              – illicit trafficking in narcotic drugs or psychotropic substances,
              – illicit trafficking in weapons, munitions or explosives,
              – murder, grievous bodily injury,
              – illicit trade in human organs or tissue,
              – illicit trafficking in nuclear or radioactive materials,
              – kidnapping, illegal restraint or hostage-taking
              – crimes within the jurisdiction of the International Criminal Court,
              – unlawful seizure of aircraft or ships,
              – rape,
              – environmental crime,
              – organised or armed robbery,
              – sabotage,
              – participation in a criminal organisation involved in one or more of the offences listed above.

        Article 6: Classification Rules for High-Risk AI Systems
              1. Irrespective of whether an AI system is placed on the market or put into service independently of the products referred to in points (a) and (b), that AI system shall be considered to be high-risk where both of the following conditions are fulfilled:
                  (a) the AI system is intended to be used as a safety component of a product, or the AI system is itself a product, covered by the Union harmonisation legislation listed in Annex I;
                  (b) the product whose safety component pursuant to point (a) is the AI system, or the AI system itself as a product, is required to undergo a third-party conformity assessment, with a view to the placing on the market or the putting into service of that product pursuant to the Union harmonisation legislation listed in Annex I.
              Related: Recitals 47, 50, and 51

              2. In addition to the high-risk AI systems referred to in paragraph 1, AI systems referred to in Annex III shall be considered to be high-risk. Related: Recitals 48, 52, 54, 55, 56, 57, 58, 59, 60, 61, 62, and 63
              3. By derogation from paragraph 2, an AI system referred to in Annex III shall not be considered to be high-risk where it does not pose a significant risk of harm to the health, safety or fundamental rights of natural persons, including by not materially influencing the outcome of decision making. The first subparagraph shall apply where any of the following conditions is fulfilled:
                  (a) the AI system is intended to perform a narrow procedural task;
                  (b) the AI system is intended to improve the result of a previously completed human activity;
                  (c) the AI system is intended to detect decision-making patterns or deviations from prior decision-making patterns and is not meant to replace or influence the previously completed human assessment, without proper human review; or
                  (d) the AI system is intended to perform a preparatory task to an assessment relevant for the purposes of the use cases listed in Annex III.
              Notwithstanding the first subparagraph, an AI system referred to in Annex III shall always be considered to be high-risk where the AI system performs profiling of natural persons.

              4. A provider who considers that an AI system referred to in Annex III is not high-risk shall document its assessment before that system is placed on the market or put into service. Such provider shall be subject to the registration obligation set out in Article 49(2). Upon request of national competent authorities, the provider shall provide the documentation of the assessment. Related: Recital 53
              5. The Commission shall, after consulting the European Artificial Intelligence Board (the ‘Board’), and no later than 2 February 2026, provide guidelines specifying the practical implementation of this Article in line with Article 96 together with a comprehensive list of practical examples of use cases of AI systems that are high-risk and not high-risk. Related: Recital 53
              6. The Commission is empowered to adopt delegated acts in accordance with Article 97 in order to amend paragraph 3, second subparagraph, of this Article by adding new conditions to those laid down therein, or by modifying them, where there is concrete and reliable evidence of the existence of AI systems that fall under the scope of Annex III, but do not pose a significant risk of harm to the health, safety or fundamental rights of natural persons. Related: Recital 53
              7. The Commission shall adopt delegated acts in accordance with Article 97 in order to amend paragraph 3, second subparagraph, of this Article by deleting any of the conditions laid down therein, where there is concrete and reliable evidence that this is necessary to maintain the level of protection of health, safety and fundamental rights provided for by this Regulation. Related: Recital 53
              8. Any amendment to the conditions laid down in paragraph 3, second subparagraph, adopted in accordance with paragraphs 6 and 7 of this Article shall not decrease the overall level of protection of health, safety and fundamental rights provided for by this Regulation and shall ensure consistency with the delegated acts adopted pursuant to Article 7(1), and take account of market and technological developments. Related: Recital 53


        Annex III: High-Risk AI Systems Referred to in Article 6(2)
            High-risk AI systems pursuant to Article 6(2) are the AI systems listed in any of the following areas:
                    1. Biometrics, in so far as their use is permitted under relevant Union or national law:
                          (a) remote biometric identification systems. This shall not include AI systems intended to be used for biometric verification the sole purpose of which is to confirm that a specific natural person is the person he or she claims to be;
                          (b) AI systems intended to be used for biometric categorisation, according to sensitive or protected attributes or characteristics based on the inference of those attributes or characteristics;
                          (c) AI systems intended to be used for emotion recognition.
                    Related: Recital 54 and Recital 159

                    2. Critical infrastructure: AI systems intended to be used as safety components in the management and operation of critical digital infrastructure, road traffic, or in the supply of water, gas, heating or electricity. Related: Recital 55
                    3. Education and vocational training:
                          (a) AI systems intended to be used to determine access or admission or to assign natural persons to educational and vocational training institutions at all levels;
                          (b) AI systems intended to be used to evaluate learning outcomes, including when those outcomes are used to steer the learning process of natural persons in educational and vocational training institutions at all levels;
                          (c) AI systems intended to be used for the purpose of assessing the appropriate level of education that an individual will receive or will be able to access, in the context of or within educational and vocational training institutions at all levels;
                          (d) AI systems intended to be used for monitoring and detecting prohibited behaviour of students during tests in the context of or within educational and vocational training institutions at all levels.
                    Related: Recital 56

                    4. Employment, workers management and access to self-employment:
                          (a) AI systems intended to be used for the recruitment or selection of natural persons, in particular to place targeted job advertisements, to analyse and filter job applications, and to evaluate candidates;
                          (b) AI systems intended to be used to make decisions affecting terms of work-related relationships, the promotion or termination of work-related contractual relationships, to allocate tasks based on individual behaviour or personal traits or characteristics or to monitor and evaluate the performance and behaviour of persons in such relationships.
                    Related: Recital 57

                    5. Access to and enjoyment of essential private services and essential public services and benefits:
                          (a) AI systems intended to be used by public authorities or on behalf of public authorities to evaluate the eligibility of natural persons for essential public assistance benefits and services, including healthcare services, as well as to grant, reduce, revoke, or reclaim such benefits and services;
                          (b) AI systems intended to be used to evaluate the creditworthiness of natural persons or establish their credit score, with the exception of AI systems used for the purpose of detecting financial fraud;
                          (c) AI systems intended to be used for risk assessment and pricing in relation to natural persons in the case of life and health insurance;
                          (d) AI systems intended to evaluate and classify emergency calls by natural persons or to be used to dispatch, or to establish priority in the dispatching of, emergency first response services, including by police, firefighters and medical aid, as well as of emergency healthcare patient triage systems.
                    Related: Recital 58

                    6. Law enforcement, in so far as their use is permitted under relevant Union or national law:
                          (a) AI systems intended to be used by or on behalf of law enforcement authorities, or by Union institutions, bodies, offices or agencies in support of law enforcement authorities or on their behalf to assess the risk of a natural person becoming the victim of criminal offences;
                          (b) AI systems intended to be used by or on behalf of law enforcement authorities or by Union institutions, bodies, offices or agencies in support of law enforcement authorities as polygraphs or similar tools;
                          (c) AI systems intended to be used by or on behalf of law enforcement authorities, or by Union institutions, bodies, offices or agencies, in support of law enforcement authorities to evaluate the reliability of evidence in the course of the investigation or prosecution of criminal offences;
                          (d) AI systems intended to be used by law enforcement authorities or on their behalf or by Union institutions, bodies, offices or agencies in support of law enforcement authorities for assessing the risk of a natural person offending or re-offending not solely on the basis of the profiling of natural persons as referred to in Article 3(4) of Directive (EU) 2016/680, or to assess personality traits and characteristics or past criminal behaviour of natural persons or groups;
                          (e) AI systems intended to be used by or on behalf of law enforcement authorities or by Union institutions, bodies, offices or agencies in support of law enforcement authorities for the profiling of natural persons as referred to in Article 3(4) of Directive (EU) 2016/680 in the course of the detection, investigation or prosecution of criminal offences.
                    Related: Recital 59

                    7. Migration, asylum and border control management, in so far as their use is permitted under relevant Union or national law:
                          (a) AI systems intended to be used by or on behalf of competent public authorities or by Union institutions, bodies, offices or agencies as polygraphs or similar tools;
                          (b) AI systems intended to be used by or on behalf of competent public authorities or by Union institutions, bodies, offices or agencies to assess a risk, including a security risk, a risk of irregular migration, or a health risk, posed by a natural person who intends to enter or who has entered into the territory of a Member State;
                          (c) AI systems intended to be used by or on behalf of competent public authorities or by Union institutions, bodies, offices or agencies to assist competent public authorities for the examination of applications for asylum, visa or residence permits and for associated complaints with regard to the eligibility of the natural persons applying for a status, including related assessments of the reliability of evidence;
                          (d) AI systems intended to be used by or on behalf of competent public authorities, or by Union institutions, bodies, offices or agencies, in the context of migration, asylum or border control management, for the purpose of detecting, recognising or identifying natural persons, with the exception of the verification of travel documents.
                    Related: Recital 60

                    8. Administration of justice and democratic processes:
                          (a) AI systems intended to be used by a judicial authority or on their behalf to assist a judicial authority in researching and interpreting facts and the law and in applying the law to a concrete set of facts, or to be used in a similar way in alternative dispute resolution; Related: Recital 61
                          (b) AI systems intended to be used for influencing the outcome of an election or referendum or the voting behaviour of natural persons in the exercise of their vote in elections or referenda. This does not include AI systems to the output of which natural persons are not directly exposed, such as tools used to organise, optimise or structure political campaigns from an administrative or logistical point of view. Related: Recital 62



        Article 7: Amendments to Annex III
           1. The Commission is empowered to adopt delegated acts in accordance with Article 97 to amend Annex III by adding or modifying use-cases of high-risk AI systems where both of the following conditions are fulfilled:
                (a) the AI systems are intended to be used in any of the areas listed in Annex III;
                (b) the AI systems pose a risk of harm to health and safety, or an adverse impact on fundamental rights, and that risk is equivalent to, or greater than, the risk of harm or of adverse impact posed by the high-risk AI systems already referred to in Annex III.

            2. When assessing the condition under paragraph 1, point (b), the Commission shall take into account the following criteria:
                (a) the intended purpose of the AI system;
                (b) the extent to which an AI system has been used or is likely to be used;
                (c) the nature and amount of the data processed and used by the AI system, in particular whether special categories of personal data are processed;
                (d) the extent to which the AI system acts autonomously and the possibility for a human to override a decision or recommendations that may lead to potential harm;
                (e) the extent to which the use of an AI system has already caused harm to health and safety, has had an adverse impact on fundamental rights or has given rise to significant concerns in relation to the likelihood of such harm or adverse impact, as demonstrated, for example, by reports or documented allegations submitted to national competent authorities or by other reports, as appropriate;
                (f) the potential extent of such harm or such adverse impact, in particular in terms of its intensity and its ability to affect multiple persons or to disproportionately affect a particular group of persons;
                (g) the extent to which persons who are potentially harmed or suffer an adverse impact are dependent on the outcome produced with an AI system, in particular because for practical or legal reasons it is not reasonably possible to opt-out from that outcome;
                (h) the extent to which there is an imbalance of power, or the persons who are potentially harmed or suffer an adverse impact are in a vulnerable position in relation to the deployer of an AI system, in particular due to status, authority, knowledge, economic or social circumstances, or age;
                (i) the extent to which the outcome produced involving an AI system is easily corrigible or reversible, taking into account the technical solutions available to correct or reverse it, whereby outcomes having an adverse impact on health, safety or fundamental rights, shall not be considered to be easily corrigible or reversible;
                (j) the magnitude and likelihood of benefit of the deployment of the AI system for individuals, groups, or society at large, including possible improvements in product safety;
                (k) the extent to which existing Union law provides for:
                      (i) effective measures of redress in relation to the risks posed by an AI system, with the exclusion of claims for damages;
                      (ii) effective measures to prevent or substantially minimise those risks.

            3. The Commission is empowered to adopt delegated acts in accordance with Article 97 to amend the list in Annex III by removing high-risk AI systems where both of the following conditions are fulfilled:
                (a) the high-risk AI system concerned no longer poses any significant risks to fundamental rights, health or safety, taking into account the criteria listed in paragraph 2;
                (b) the deletion does not decrease the overall level of protection of health, safety and fundamental rights under Union law.



        Write the reasoning to be concise, fitting into one sentence of a maximum of 125 words and distilling the key info, which helps to understand why the use is Prohibited, High Risk or Limited or Low Risk.
        The format for the statement is as follows:
        if the use is classified as "Prohibited/High Risk/Excluded/High-Risk Exception":
            "Prohibited/High Risk/Excluded/High-Risk Exception" due to [THE REASON], which falls under the EU AI Act [RELEVANT SECTION or RULE].
        or, if the classification is "Limited or Low Risk":
            "Limited or Low Risk" due to [THE REASON] [mention the EU AI Act if useful only].

        For example, Biometric categorization of people by Law enforcement agencies is Prohibited according to the Article 5 (g).
        Here are the details of the AI system:

        Domain: {{ domain }},
        Purpose: {{ purpose }},
        Capability: {{ capability }},
        AI User: {{ ai_user }},
        AI Subject: {{ ai_subject }}

         Please return the classification in the following format:
         {
           "Description": "The AI system intended to be used ...",
           "Classification": "Excluded"/"Prohibited"/"High-Risk Exception"/"High Risk"/"Limited or Low Risk",
           "AIActText": "[Quotation if applicable] - Include the amendment or EU AI Act section that mostly closely resembles the text.",
           "Reasoning": "[Explanation]"
         }
"""
