"""Integration tests for benchmark framework."""

import sys
import os
import numpy as np
import pandas as pd

# Add dashboard to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from metrics.pairwise import compute_pairwise_metrics
from metrics.clustering import compute_clustering_metrics, validate_clustering_inputs
from metrics.ranking import compute_ranking_metrics
from calibration import FixedThresholdCalibrator, PlattCalibrator, IsotonicCalibrator
from clustering import ConnectedComponentsClusterer, HACClusterer
from blocking import TrivialBlocking, TokenBlocking


def test_pairwise_metrics():
    """Test pairwise metrics computation."""
    print("Testing pairwise metrics...")
    
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_pred = np.array([1, 0, 0, 1, 1, 0])
    y_scores = np.array([0.9, 0.4, 0.1, 0.6, 0.8, 0.2])
    
    metrics = compute_pairwise_metrics(y_true, y_pred, y_scores)
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "accuracy" in metrics
    assert "auc" in metrics
    
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1: {metrics['f1']:.3f}")
    print("  ✓ Pairwise metrics test passed")


def test_clustering_metrics():
    """Test clustering metrics with validation."""
    print("\nTesting clustering metrics...")
    
    # Valid clusters
    pred_clusters = [0, 0, 1, 1, 2, 2, -1]
    true_clusters = [0, 0, 0, 1, 1, 1, -1]
    
    # Validate inputs
    is_valid, issues = validate_clustering_inputs(pred_clusters, true_clusters)
    assert is_valid, f"Validation failed: {issues}"
    print("  ✓ Validation passed")
    
    # Compute metrics
    metrics = compute_clustering_metrics(pred_clusters, true_clusters)
    
    assert "ari" in metrics
    assert "nmi" in metrics
    assert "b3_precision" in metrics
    assert "b3_recall" in metrics
    assert "b3_f1" in metrics
    
    print(f"  ARI: {metrics['ari']:.3f}")
    print(f"  NMI: {metrics['nmi']:.3f}")
    print(f"  B³ F1: {metrics['b3_f1']:.3f}")
    print("  ✓ Clustering metrics test passed")


def test_ranking_metrics():
    """Test ranking metrics."""
    print("\nTesting ranking metrics...")
    
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0])
    y_scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
    
    metrics = compute_ranking_metrics(y_true, y_scores, k_values=[3, 5])
    
    assert "p_at_3" in metrics
    assert "r_at_3" in metrics
    assert "map" in metrics
    
    print(f"  P@3: {metrics['p_at_3']:.3f}")
    print(f"  R@3: {metrics['r_at_3']:.3f}")
    print(f"  MAP: {metrics['map']:.3f}")
    print("  ✓ Ranking metrics test passed")


def test_calibrators():
    """Test calibration methods."""
    print("\nTesting calibrators...")
    
    val_scores = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9])
    val_labels = np.array([0, 0, 0, 1, 1, 1, 1])
    test_scores = np.array([0.4, 0.5, 0.75])
    
    # Fixed threshold
    fixed = FixedThresholdCalibrator(threshold=0.5)
    fixed.fit(val_scores, val_labels)
    preds = fixed.predict(test_scores)
    print(f"  Fixed threshold predictions: {preds}")
    
    # Platt scaling
    platt = PlattCalibrator()
    platt.fit(val_scores, val_labels)
    preds = platt.predict(test_scores)
    print(f"  Platt predictions: {preds}")
    
    # Isotonic
    isotonic = IsotonicCalibrator()
    isotonic.fit(val_scores, val_labels)
    preds = isotonic.predict(test_scores)
    print(f"  Isotonic predictions: {preds}")
    
    print("  ✓ Calibrators test passed")


def test_clustering_algorithms():
    """Test clustering algorithms."""
    print("\nTesting clustering algorithms...")
    
    # Create sample edges
    edges = pd.DataFrame({
        "id1": [1, 1, 2, 4, 4],
        "id2": [2, 3, 3, 5, 6],
        "score": [0.9, 0.8, 0.85, 0.95, 0.7]
    })
    record_ids = [1, 2, 3, 4, 5, 6, 7]
    
    # Connected components
    cc = ConnectedComponentsClusterer(threshold=0.7)
    clusters = cc.cluster(edges, record_ids=record_ids)
    print(f"  CC clusters: {clusters}")
    assert len(clusters) == len(record_ids)
    
    # HAC
    hac = HACClusterer(distance_threshold=0.3)
    clusters = hac.cluster(edges, record_ids=record_ids)
    print(f"  HAC clusters: {clusters}")
    assert len(clusters) == len(record_ids)
    
    print("  ✓ Clustering algorithms test passed")


def test_blocking():
    """Test blocking strategies."""
    print("\nTesting blocking strategies...")
    
    # Create sample records
    records = pd.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "name": ["John Smith", "Jon Smith", "Jane Doe", "John Doe", "Bob Jones"],
        "text": ["John Smith 30", "Jon Smith 31", "Jane Doe 25", "John Doe 28", "Bob Jones 40"]
    })
    
    # Trivial blocking (small dataset)
    trivial = TrivialBlocking(max_records=10)
    pairs = trivial.generate_candidates(records)
    print(f"  Trivial blocking: {len(pairs)} pairs")
    assert len(pairs) == 10  # 5 choose 2
    
    # Token blocking
    token = TokenBlocking(blocking_fields=["name"])
    pairs = token.generate_candidates(records)
    print(f"  Token blocking: {len(pairs)} pairs")
    assert len(pairs) > 0
    
    print("  ✓ Blocking strategies test passed")


def run_all_tests():
    """Run all integration tests."""
    print("="*60)
    print("Running Integration Tests")
    print("="*60)
    
    try:
        test_pairwise_metrics()
        test_clustering_metrics()
        test_ranking_metrics()
        test_calibrators()
        test_clustering_algorithms()
        test_blocking()
        
        print("\n" + "="*60)
        print("✓ All tests passed!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

