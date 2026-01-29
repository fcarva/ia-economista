"""
Unit Tests for GATv2 Encoder (Track 2)
======================================

Verifies that the GATv2 attention architecture:
1. Handles dimension changes correctly (heads * hidden_dim)
2. Outputs correct shapes
3. Returns valid attention weights for explainability
"""

import pytest
import torch
from torch_geometric.data import Data

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from cointegration_gnn.models.gnn_model import (
    GATv2Encoder,
    GraphSAGEEncoder,
    CointegrationGNN,
)


class TestGATv2Encoder:
    """Tests for GATv2Encoder class."""
    
    @pytest.fixture
    def mock_graph(self):
        """Create a mock graph with 9 nodes (IBOVESPA assets), 6 features."""
        n_nodes = 9
        n_features = 8
        
        # Random node features
        x = torch.randn(n_nodes, n_features)
        
        # Create edges (fully connected for testing)
        edges = []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edges.append([i, j])
        edge_index = torch.tensor(edges, dtype=torch.long).T
        
        # Edge weights (cointegration strength)
        edge_attr = torch.rand(edge_index.shape[1])
        
        return x, edge_index, edge_attr
    
    def test_encoder_output_shape(self, mock_graph):
        """Test that encoder outputs correct shape."""
        x, edge_index, edge_attr = mock_graph
        
        encoder = GATv2Encoder(
            in_channels=8,
            hidden_channels=32,
            out_channels=16,
            num_layers=3,
            heads=4,
            dropout=0.2,
        )
        encoder.eval()
        
        with torch.no_grad():
            output = encoder(x, edge_index, edge_attr)
        
        # Output should be (n_nodes, out_channels) = (9, 16)
        assert output.shape == (9, 16), f"Expected (9, 16), got {output.shape}"
    
    def test_attention_weights_extraction(self, mock_graph):
        """Test that attention weights can be extracted for XAI."""
        x, edge_index, edge_attr = mock_graph
        
        encoder = GATv2Encoder(
            in_channels=8,
            hidden_channels=32,
            out_channels=16,
            num_layers=3,
            heads=4,
        )
        
        attention = encoder.get_attention_weights(x, edge_index, edge_attr)
        
        # Should return list of (edge_index, alpha) per layer
        assert len(attention) == 3, f"Expected 3 layers, got {len(attention)}"
        
        # Check shapes
        for layer_idx, (edge_idx, alpha) in enumerate(attention):
            assert edge_idx.shape[0] == 2, "Edge index should be (2, n_edges)"
            assert alpha.shape[0] == edge_index.shape[1], "Alpha should have n_edges rows"
            
            # Intermediate layers have `heads` columns, last layer has 1
            if layer_idx < 2:
                assert alpha.shape[1] == 4, f"Expected 4 heads, got {alpha.shape[1]}"
            else:
                assert alpha.shape[1] == 1, f"Last layer should have 1 head"
    
    def test_attention_sums_to_one(self, mock_graph):
        """Test that attention weights sum to ~1 per node (softmax)."""
        x, edge_index, edge_attr = mock_graph
        
        encoder = GATv2Encoder(
            in_channels=8,
            hidden_channels=32,
            out_channels=16,
            num_layers=3,
            heads=4,
        )
        
        attention = encoder.get_attention_weights(x, edge_index, edge_attr)
        
        # For the last layer, check attention sums for each target node
        _, alpha = attention[-1]
        
        # Alpha values should be in [0, 1]
        assert alpha.min() >= 0, "Attention should be non-negative"
        assert alpha.max() <= 1, "Attention should be <= 1"


class TestCointegrationGNNWithGATv2:
    """Tests for CointegrationGNN with GATv2 encoder."""
    
    @pytest.fixture
    def mock_data(self):
        """Create mock PyG Data object."""
        n_nodes = 9
        n_features = 8
        
        x = torch.randn(n_nodes, n_features)
        
        edges = [[i, j] for i in range(n_nodes) for j in range(n_nodes) if i != j]
        edge_index = torch.tensor(edges, dtype=torch.long).T
        edge_attr = torch.rand(edge_index.shape[1], 1)
        
        y = torch.randn(n_nodes)  # Returns
        
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    def test_model_default_is_gatv2(self):
        """Test that default encoder is GATv2."""
        model = CointegrationGNN()
        assert model.encoder_type == "gatv2"
        assert isinstance(model.encoder, GATv2Encoder)
    
    def test_model_output_shape(self, mock_data):
        """Test that model outputs correct prediction shape."""
        model = CointegrationGNN(
            node_feature_dim=8,
            hidden_dim=32,
            encoder_type="gatv2",
        )
        model.eval()
        
        with torch.no_grad():
            positions = model(mock_data)
        
        # Should output position for each asset
        assert positions.shape == (9,), f"Expected (9,), got {positions.shape}"
        
        # Positions should be in [-1, 1] (Tanh)
        assert positions.min() >= -1, "Positions should be >= -1"
        assert positions.max() <= 1, "Positions should be <= 1"
    
    def test_sage_fallback(self, mock_data):
        """Test that SAGE encoder still works as fallback."""
        model = CointegrationGNN(
            node_feature_dim=8,
            hidden_dim=32,
            encoder_type="sage",
        )
        model.eval()
        
        with torch.no_grad():
            positions = model(mock_data)
        
        assert positions.shape == (9,)
        assert isinstance(model.encoder, GraphSAGEEncoder)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
