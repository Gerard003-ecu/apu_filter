import networkx as nx
from app.tactics.logistics_manifold import LogisticsManifold
from app.core.mic_algebra import CategoricalState
import numpy as np
G = nx.DiGraph()
G.add_edge("A", "B", flow=1.0)
G.nodes["A"]["sink_source"] = 1.0
G.nodes["B"]["sink_source"] = -1.0
G.nodes["A"]["delay"] = 5.0
state = CategoricalState(payload={}, context={'logistics_graph': G}, validated_strata=frozenset(), error=None, error_details=None)
lm = LogisticsManifold()
res = lm(state)
print(res)
if res.is_success:
  g_res = res.context['logistics_graph']
  print("effective mass:", g_res.nodes["A"].get("effective_mass"))
