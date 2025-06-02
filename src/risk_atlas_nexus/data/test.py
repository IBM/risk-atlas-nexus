risk_id = None

evaluation_instances = [{"hasRelatedRisk": ["ok"]}]

evaluation_instances = list(
    filter(
        lambda evaluation: risk_id in evaluation["hasRelatedRisk"],
        evaluation_instances,
    )
)

print(evaluation_instances)
