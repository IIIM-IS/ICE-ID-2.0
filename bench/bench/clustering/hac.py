"""Hierarchical Agglomerative Clustering."""

from typing import List, Tuple, Dict
from collections import defaultdict
import numpy as np

from .base import BaseClusterer
from ..core.types import ClusterResult
from ..core.registry import get_registry


class HACClusterer(BaseClusterer):
    """
    Hierarchical Agglomerative Clustering.
    
    Uses average linkage on pairwise scores.
    """
    
    def __init__(
        self,
        threshold: float = 0.5,
        linkage: str = "average",
        **kwargs
    ):
        """
        Initialize HAC clusterer.
        
        Args:
            threshold: Distance threshold for cutting dendrogram.
            linkage: Linkage method (average, single, complete).
        """
        super().__init__("hac", **kwargs)
        self.threshold = threshold
        self.linkage = linkage
    
    def cluster(
        self,
        edges: List[Tuple[int, int, float]],
        all_record_ids: List[int]
    ) -> ClusterResult:
        """Perform HAC clustering."""
        if not edges:
            return ClusterResult(clusters=[[r] for r in all_record_ids])
        
        nodes = set(all_record_ids)
        for id1, id2, _ in edges:
            nodes.add(id1)
            nodes.add(id2)
        
        node_to_idx = {n: i for i, n in enumerate(sorted(nodes))}
        idx_to_node = {i: n for n, i in node_to_idx.items()}
        n = len(nodes)
        
        if n < 2:
            return ClusterResult(clusters=[list(nodes)])
        
        dist_matrix = np.ones((n, n))
        np.fill_diagonal(dist_matrix, 0)
        
        for id1, id2, score in edges:
            i, j = node_to_idx[id1], node_to_idx[id2]
            dist = 1 - score
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
        
        clusters_map = {i: [i] for i in range(n)}
        active = set(range(n))
        
        while len(active) > 1:
            min_dist = float("inf")
            merge_pair = None
            
            active_list = sorted(active)
            for i_idx in range(len(active_list)):
                for j_idx in range(i_idx + 1, len(active_list)):
                    i, j = active_list[i_idx], active_list[j_idx]
                    
                    if self.linkage == "average":
                        dist = 0
                        count = 0
                        for m1 in clusters_map[i]:
                            for m2 in clusters_map[j]:
                                dist += dist_matrix[m1, m2]
                                count += 1
                        dist = dist / count if count > 0 else 1.0
                    elif self.linkage == "single":
                        dist = min(
                            dist_matrix[m1, m2]
                            for m1 in clusters_map[i]
                            for m2 in clusters_map[j]
                        )
                    else:
                        dist = max(
                            dist_matrix[m1, m2]
                            for m1 in clusters_map[i]
                            for m2 in clusters_map[j]
                        )
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_pair = (i, j)
            
            if merge_pair is None or min_dist > self.threshold:
                break
            
            i, j = merge_pair
            clusters_map[i].extend(clusters_map[j])
            del clusters_map[j]
            active.remove(j)
        
        clusters = []
        for c_idx in active:
            cluster = [idx_to_node[idx] for idx in clusters_map[c_idx]]
            clusters.append(sorted(cluster))
        
        return ClusterResult(clusters=clusters)


get_registry("clusterers").register("hac", HACClusterer)

