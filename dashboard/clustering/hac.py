"""Hierarchical agglomerative clustering."""

from typing import List
import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from .base import Clusterer


class HACClusterer(Clusterer):
    """
    Hierarchical Agglomerative Clustering (HAC).
    
    Uses average linkage by default and cuts dendrogram at a threshold.
    """
    
    def __init__(
        self,
        linkage_method: str = "average",
        distance_threshold: float = 0.5,
    ):
        """
        Initialize HAC clusterer.
        
        Args:
            linkage_method: Linkage method ("average", "complete", "single")
            distance_threshold: Threshold for cutting dendrogram
        """
        self.linkage_method = linkage_method
        self.distance_threshold = distance_threshold
    
    def cluster(
        self,
        edges: pd.DataFrame = None,
        scores: np.ndarray = None,
        record_ids: List[int] = None,
    ) -> List[int]:
        """
        Cluster using hierarchical agglomerative clustering.
        
        Args:
            edges: DataFrame with columns [id1, id2, score/weight]
            scores: Alternative: full pairwise similarity matrix
            record_ids: List of all record IDs
            
        Returns:
            List of cluster assignments
        """
        if edges is None and scores is None:
            return [-1] * len(record_ids) if record_ids else []
        
        # Build distance matrix from edges or scores
        if scores is not None:
            # Scores provided as matrix
            distances = 1 - scores  # Convert similarity to distance
        else:
            # Build distance matrix from edges
            if record_ids is None:
                # Extract unique IDs from edges
                record_ids = sorted(set(
                    list(edges["id1"].unique()) + list(edges["id2"].unique())
                ))
            
            n = len(record_ids)
            id_to_idx = {rid: idx for idx, rid in enumerate(record_ids)}
            
            # Initialize with maximum distance
            distances = np.ones((n, n))
            np.fill_diagonal(distances, 0)
            
            # Fill in edges
            for _, row in edges.iterrows():
                id1, id2 = int(row["id1"]), int(row["id2"])
                
                if id1 not in id_to_idx or id2 not in id_to_idx:
                    continue
                
                idx1, idx2 = id_to_idx[id1], id_to_idx[id2]
                
                # Get similarity score (default 1.0 if not present)
                similarity = 1.0
                for col in ["score", "weight", "similarity"]:
                    if col in row:
                        similarity = float(row[col])
                        break
                
                distance = 1 - similarity
                distances[idx1, idx2] = distance
                distances[idx2, idx1] = distance
        
        # Ensure distances are valid
        if np.any(np.isnan(distances)) or np.any(np.isinf(distances)):
            distances = np.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=0.0)
        
        # Perform hierarchical clustering
        try:
            # Convert to condensed distance matrix
            condensed_dist = squareform(distances, checks=False)
            
            # Compute linkage
            Z = linkage(condensed_dist, method=self.linkage_method)
            
            # Cut dendrogram
            cluster_labels = fcluster(
                Z,
                t=self.distance_threshold,
                criterion='distance'
            )
            
            # Convert to 0-indexed
            clusters = [int(c) - 1 for c in cluster_labels]
            
        except Exception as e:
            print(f"Warning: HAC clustering failed: {e}")
            # Fallback: each record is its own cluster
            clusters = list(range(len(record_ids))) if record_ids else []
        
        return clusters

