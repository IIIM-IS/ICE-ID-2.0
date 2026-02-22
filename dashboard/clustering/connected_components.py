"""Connected components clustering."""

from typing import List, Dict, Set
import pandas as pd
import numpy as np
from collections import defaultdict, deque
from .base import Clusterer


class ConnectedComponentsClusterer(Clusterer):
    """
    Connected components clustering.
    
    Treats edges as an undirected graph and finds connected components.
    Each component becomes a cluster.
    """
    
    def __init__(self, threshold: float = None):
        """
        Initialize connected components clusterer.
        
        Args:
            threshold: Optional threshold for edge weights (if provided in edges)
        """
        self.threshold = threshold
    
    def cluster(
        self,
        edges: pd.DataFrame = None,
        scores: np.ndarray = None,
        record_ids: List[int] = None,
    ) -> List[int]:
        """
        Cluster using connected components.
        
        Args:
            edges: DataFrame with columns [id1, id2] and optionally [score/weight]
            scores: Not used (edges are sufficient)
            record_ids: List of all record IDs
            
        Returns:
            List of cluster assignments
        """
        if edges is None or len(edges) == 0:
            # No edges - each record is its own cluster
            return [-1] * len(record_ids) if record_ids else []
        
        # Filter edges by threshold if weight column exists
        if self.threshold is not None:
            weight_col = None
            for col in ["score", "weight", "similarity"]:
                if col in edges.columns:
                    weight_col = col
                    break
            
            if weight_col:
                edges = edges[edges[weight_col] >= self.threshold].copy()
        
        # Build adjacency list
        adj: Dict[int, Set[int]] = defaultdict(set)
        nodes_in_edges = set()
        
        for _, row in edges.iterrows():
            id1 = int(row["id1"])
            id2 = int(row["id2"])
            
            adj[id1].add(id2)
            adj[id2].add(id1)
            nodes_in_edges.add(id1)
            nodes_in_edges.add(id2)
        
        # Find connected components using BFS
        visited = set()
        cluster_id = 0
        node_to_cluster: Dict[int, int] = {}
        
        def bfs(start: int) -> List[int]:
            """BFS to find connected component."""
            queue = deque([start])
            visited.add(start)
            component = []
            
            while queue:
                node = queue.popleft()
                component.append(node)
                
                for neighbor in adj[node]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            
            return component
        
        # Assign cluster IDs to connected components
        for node in nodes_in_edges:
            if node not in visited:
                component = bfs(node)
                for n in component:
                    node_to_cluster[n] = cluster_id
                cluster_id += 1
        
        # Map record IDs to clusters
        if record_ids is None:
            record_ids = sorted(nodes_in_edges)
        
        clusters = []
        for rid in record_ids:
            if rid in node_to_cluster:
                clusters.append(node_to_cluster[rid])
            else:
                clusters.append(-1)  # Singleton (no edges)
        
        return clusters

