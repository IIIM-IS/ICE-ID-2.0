"""Connected components clustering."""

from typing import List, Tuple
from collections import defaultdict

from .base import BaseClusterer
from ..core.types import ClusterResult
from ..core.registry import get_registry


class ConnectedComponentsClusterer(BaseClusterer):
    """
    Connected components clustering.
    
    Groups records that are transitively connected through edges.
    """
    
    def __init__(self, **kwargs):
        super().__init__("connected_components", **kwargs)
    
    def cluster(
        self,
        edges: List[Tuple[int, int, float]],
        all_record_ids: List[int]
    ) -> ClusterResult:
        """Find connected components."""
        graph = defaultdict(list)
        
        for id1, id2, score in edges:
            graph[id1].append(id2)
            graph[id2].append(id1)
        
        all_nodes = set(all_record_ids)
        for id1, id2, _ in edges:
            all_nodes.add(id1)
            all_nodes.add(id2)
        
        visited = set()
        clusters = []
        
        for node in sorted(all_nodes):
            if node in visited:
                continue
            
            cluster = []
            queue = [node]
            visited.add(node)
            
            while queue:
                current = queue.pop(0)
                cluster.append(current)
                
                for neighbor in graph[current]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            clusters.append(sorted(cluster))
        
        return ClusterResult(clusters=clusters)


get_registry("clusterers").register("connected_components", ConnectedComponentsClusterer)

