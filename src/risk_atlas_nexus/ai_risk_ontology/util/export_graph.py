# Convenience script to export a loaded graph from all the yaml
# in src/risk_atlas_nexus/data/knowledge_graph

from risk_atlas_nexus import RiskAtlasNexus


ran = RiskAtlasNexus()

# export the graph to yaml
ran.export("graph_export/yaml/")
