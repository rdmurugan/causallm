"""
Causal Foundation Models for Tier 3 capabilities.

This module implements pre-trained causal reasoning models, causal representation learning,
and foundation model architectures specifically designed for causal inference tasks.
It provides the core infrastructure for advanced AI-native causal reasoning.
"""

from typing import Dict, List, Optional, Tuple, Any, Union, Callable, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import asyncio
import json
import pickle
from pathlib import Path
import hashlib
from datetime import datetime
import warnings

from causalllm.logging import get_logger


class CausalModelType(Enum):
    """Types of causal foundation models."""
    GRAPH_ENCODER = "graph_encoder"                    # Encode causal graphs
    VARIABLE_ENCODER = "variable_encoder"              # Encode variable relationships  
    INTERVENTION_PREDICTOR = "intervention_predictor"  # Predict intervention effects
    COUNTERFACTUAL_GENERATOR = "counterfactual_generator"  # Generate counterfactuals
    DISCOVERY_MODEL = "discovery_model"                # Discover causal structures
    EXPLANATION_MODEL = "explanation_model"            # Generate causal explanations
    UNIFIED_CAUSAL = "unified_causal"                  # Multi-task causal model


class CausalModelArchitecture(Enum):
    """Foundation model architectures for causal reasoning."""
    TRANSFORMER = "transformer"
    GRAPH_NEURAL_NETWORK = "gnn"
    VARIATIONAL_AUTOENCODER = "vae"
    DIFFUSION_MODEL = "diffusion"
    HYBRID_NEURAL_SYMBOLIC = "hybrid_ns"
    MEMORY_AUGMENTED = "memory_augmented"


class TrainingObjective(Enum):
    """Training objectives for causal foundation models."""
    CAUSAL_DISCOVERY = "causal_discovery"
    INTERVENTION_PREDICTION = "intervention_prediction"
    COUNTERFACTUAL_INFERENCE = "counterfactual_inference"
    GRAPH_RECONSTRUCTION = "graph_reconstruction"
    VARIABLE_DISENTANGLEMENT = "variable_disentanglement"
    MULTI_TASK_CAUSAL = "multi_task_causal"


@dataclass
class CausalRepresentation:
    """Learned causal representation of a system."""
    
    variable_embeddings: torch.Tensor
    graph_embedding: torch.Tensor
    causal_mechanisms: Dict[str, torch.Tensor]
    intervention_embeddings: Dict[str, torch.Tensor]
    latent_confounders: Optional[torch.Tensor] = None
    uncertainty_estimates: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CausalModelConfig:
    """Configuration for causal foundation models."""
    
    model_type: CausalModelType
    architecture: CausalModelArchitecture
    hidden_dim: int = 512
    num_layers: int = 6
    num_heads: int = 8
    dropout_rate: float = 0.1
    max_variables: int = 100
    max_sequence_length: int = 1024
    use_positional_encoding: bool = True
    use_graph_attention: bool = True
    causal_masking: bool = True
    temperature: float = 1.0
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    training_objectives: List[TrainingObjective] = field(default_factory=lambda: [TrainingObjective.MULTI_TASK_CAUSAL])


@dataclass
class CausalTrainingExample:
    """Training example for causal foundation models."""
    
    variables: Dict[str, Any]
    graph_edges: List[Tuple[str, str]]
    interventions: Dict[str, Any]
    outcomes: Dict[str, Any]
    counterfactuals: Dict[str, Any]
    domain: str
    context: str
    ground_truth_mechanisms: Optional[Dict[str, Any]] = None


class CausalDataset(Dataset):
    """Dataset for training causal foundation models."""
    
    def __init__(self, examples: List[CausalTrainingExample], 
                 tokenizer: Optional[Any] = None,
                 max_variables: int = 100):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_variables = max_variables
        self.variable_vocab = self._build_variable_vocabulary()
        self.logger = get_logger("causalllm.causal_foundation_models.dataset")
        
    def _build_variable_vocabulary(self) -> Dict[str, int]:
        """Build vocabulary of variable names."""
        vocab = {"<PAD>": 0, "<UNK>": 1, "<START>": 2, "<END>": 3}
        idx = 4
        
        for example in self.examples:
            for var in example.variables.keys():
                if var not in vocab:
                    vocab[var] = idx
                    idx += 1
        
        return vocab
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Encode variables
        variable_ids = []
        for var in example.variables.keys():
            var_id = self.variable_vocab.get(var, self.variable_vocab["<UNK>"])
            variable_ids.append(var_id)
        
        # Pad or truncate to max_variables
        if len(variable_ids) < self.max_variables:
            variable_ids.extend([self.variable_vocab["<PAD>"]] * (self.max_variables - len(variable_ids)))
        else:
            variable_ids = variable_ids[:self.max_variables]
        
        # Create adjacency matrix for graph
        adjacency = torch.zeros(self.max_variables, self.max_variables)
        var_to_idx = {var: i for i, var in enumerate(example.variables.keys())}
        
        for cause, effect in example.graph_edges:
            if cause in var_to_idx and effect in var_to_idx:
                cause_idx = var_to_idx[cause]
                effect_idx = var_to_idx[effect]
                if cause_idx < self.max_variables and effect_idx < self.max_variables:
                    adjacency[cause_idx, effect_idx] = 1.0
        
        return {
            "variable_ids": torch.tensor(variable_ids, dtype=torch.long),
            "adjacency_matrix": adjacency,
            "domain": example.domain,
            "context": example.context,
            "num_variables": min(len(example.variables), self.max_variables)
        }


class CausalTransformer(nn.Module):
    """Transformer architecture for causal reasoning."""
    
    def __init__(self, config: CausalModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Variable embedding layer
        self.variable_embedding = nn.Embedding(vocab_size, config.hidden_dim)
        
        # Positional encoding
        if config.use_positional_encoding:
            self.pos_encoding = nn.Parameter(torch.randn(config.max_variables, config.hidden_dim))
        
        # Transformer layers with causal masking
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_dim * 4,
                dropout=config.dropout_rate,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        
        # Graph attention mechanism
        if config.use_graph_attention:
            self.graph_attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=config.num_heads,
                dropout=config.dropout_rate,
                batch_first=True
            )
        
        # Output heads for different tasks
        self.graph_prediction_head = nn.Linear(config.hidden_dim, config.max_variables)
        self.intervention_effect_head = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.counterfactual_head = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        self.dropout = nn.Dropout(config.dropout_rate)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
    def forward(self, variable_ids: torch.Tensor, 
                adjacency_matrix: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        
        batch_size, seq_len = variable_ids.shape
        
        # Variable embeddings
        embeddings = self.variable_embedding(variable_ids)
        
        # Add positional encoding
        if self.config.use_positional_encoding:
            embeddings = embeddings + self.pos_encoding[:seq_len].unsqueeze(0)
        
        # Apply transformer layers
        hidden_states = embeddings
        for layer in self.transformer_layers:
            hidden_states = layer(hidden_states, src_key_padding_mask=attention_mask)
        
        # Graph attention if enabled
        if self.config.use_graph_attention and adjacency_matrix is not None:
            # Use adjacency matrix to create attention mask
            graph_attn_mask = (adjacency_matrix == 0)
            graph_attended, _ = self.graph_attention(
                hidden_states, hidden_states, hidden_states,
                key_padding_mask=graph_attn_mask.view(batch_size, -1)
            )
            hidden_states = hidden_states + graph_attended
        
        hidden_states = self.layer_norm(self.dropout(hidden_states))
        
        # Multi-task outputs
        outputs = {
            "hidden_states": hidden_states,
            "graph_predictions": self.graph_prediction_head(hidden_states),
            "intervention_effects": self.intervention_effect_head(hidden_states),
            "counterfactual_representations": self.counterfactual_head(hidden_states)
        }
        
        return outputs


class CausalGraphNeuralNetwork(nn.Module):
    """Graph Neural Network for causal structure learning."""
    
    def __init__(self, config: CausalModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.hidden_dim = config.hidden_dim
        
        # Node embeddings
        self.node_embedding = nn.Embedding(vocab_size, config.hidden_dim)
        
        # Graph convolutional layers
        self.conv_layers = nn.ModuleList([
            GraphConvLayer(config.hidden_dim, config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # Edge prediction
        self.edge_predictor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1),
            nn.Sigmoid()
        )
        
        # Causal mechanism encoder
        self.mechanism_encoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim)
        )
        
    def forward(self, variable_ids: torch.Tensor,
                adjacency_matrix: torch.Tensor) -> Dict[str, torch.Tensor]:
        
        batch_size, num_nodes = variable_ids.shape
        
        # Node embeddings
        node_embeddings = self.node_embedding(variable_ids)
        
        # Apply graph convolutions
        hidden_states = node_embeddings
        for conv_layer in self.conv_layers:
            hidden_states = conv_layer(hidden_states, adjacency_matrix)
        
        # Predict edges
        edge_probs = torch.zeros(batch_size, num_nodes, num_nodes)
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edge_input = torch.cat([hidden_states[:, i], hidden_states[:, j]], dim=-1)
                    edge_probs[:, i, j] = self.edge_predictor(edge_input).squeeze(-1)
        
        # Encode causal mechanisms
        mechanism_embeddings = self.mechanism_encoder(hidden_states)
        
        return {
            "node_embeddings": hidden_states,
            "edge_probabilities": edge_probs,
            "mechanism_embeddings": mechanism_embeddings
        }


class GraphConvLayer(nn.Module):
    """Graph convolution layer for causal GNN."""
    
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, node_features: torch.Tensor, 
                adjacency_matrix: torch.Tensor) -> torch.Tensor:
        
        # Graph convolution: aggregate neighbor features
        aggregated = torch.bmm(adjacency_matrix, node_features)
        
        # Transform and activate
        output = self.activation(self.linear(aggregated))
        return self.dropout(output)


class CausalFoundationModel(nn.Module):
    """Unified causal foundation model."""
    
    def __init__(self, config: CausalModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.logger = get_logger("causalllm.causal_foundation_model")
        
        # Select architecture
        if config.architecture == CausalModelArchitecture.TRANSFORMER:
            self.backbone = CausalTransformer(config, vocab_size)
        elif config.architecture == CausalModelArchitecture.GRAPH_NEURAL_NETWORK:
            self.backbone = CausalGraphNeuralNetwork(config, vocab_size)
        else:
            raise ValueError(f"Unsupported architecture: {config.architecture}")
        
        # Task-specific heads
        self.causal_discovery_head = nn.Linear(config.hidden_dim, config.max_variables)
        self.intervention_head = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.explanation_head = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass through the foundation model."""
        
        # Extract inputs
        variable_ids = batch["variable_ids"]
        adjacency_matrix = batch.get("adjacency_matrix")
        
        # Backbone forward pass
        backbone_outputs = self.backbone(variable_ids, adjacency_matrix)
        
        # Task-specific outputs
        outputs = backbone_outputs.copy()
        
        if "hidden_states" in backbone_outputs:
            hidden = backbone_outputs["hidden_states"]
            
            # Pool over sequence dimension for global representations
            pooled = hidden.mean(dim=1)  # [batch_size, hidden_dim]
            
            outputs.update({
                "causal_discovery_logits": self.causal_discovery_head(pooled),
                "intervention_representations": self.intervention_head(pooled),
                "explanation_representations": self.explanation_head(pooled)
            })
        
        return outputs
    
    def encode_causal_system(self, variables: Dict[str, str], 
                           graph_edges: List[Tuple[str, str]],
                           domain: str = "general") -> CausalRepresentation:
        """Encode a causal system into learned representations."""
        
        # Create dummy example for encoding
        example = CausalTrainingExample(
            variables=variables,
            graph_edges=graph_edges,
            interventions={},
            outcomes={},
            counterfactuals={},
            domain=domain,
            context=""
        )
        
        # Create dataset and dataloader
        dataset = CausalDataset([example], max_variables=self.config.max_variables)
        batch = dataset[0]
        
        # Add batch dimension
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                batch[key] = value.unsqueeze(0)
        
        # Forward pass
        self.eval()
        with torch.no_grad():
            outputs = self.forward(batch)
        
        # Extract representations
        representation = CausalRepresentation(
            variable_embeddings=outputs["hidden_states"].squeeze(0),
            graph_embedding=outputs["causal_discovery_logits"].squeeze(0),
            causal_mechanisms={"default": outputs["intervention_representations"].squeeze(0)},
            intervention_embeddings={"default": outputs["intervention_representations"].squeeze(0)},
            metadata={
                "domain": domain,
                "num_variables": len(variables),
                "num_edges": len(graph_edges)
            }
        )
        
        return representation


class CausalFoundationModelTrainer:
    """Trainer for causal foundation models."""
    
    def __init__(self, model: CausalFoundationModel, 
                 config: CausalModelConfig,
                 device: str = "cpu"):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.logger = get_logger("causalllm.causal_foundation_model_trainer")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Loss functions
        self.graph_loss_fn = nn.BCELoss()
        self.intervention_loss_fn = nn.MSELoss()
        self.explanation_loss_fn = nn.MSELoss()
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {"total": 0.0, "graph": 0.0, "intervention": 0.0}
        num_batches = 0
        
        for batch in dataloader:
            # Move to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(self.device)
            
            # Forward pass
            outputs = self.model(batch)
            
            # Compute losses
            losses = self._compute_losses(outputs, batch)
            total_loss = sum(losses.values())
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            # Track losses
            epoch_losses["total"] += total_loss.item()
            for loss_name, loss_value in losses.items():
                if loss_name in epoch_losses:
                    epoch_losses[loss_name] += loss_value.item()
            
            num_batches += 1
        
        # Average losses
        for loss_name in epoch_losses:
            epoch_losses[loss_name] /= num_batches
        
        return epoch_losses
    
    def _compute_losses(self, outputs: Dict[str, torch.Tensor],
                       batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute multi-task losses."""
        losses = {}
        
        # Graph structure prediction loss
        if "edge_probabilities" in outputs and "adjacency_matrix" in batch:
            predicted_edges = outputs["edge_probabilities"]
            true_edges = batch["adjacency_matrix"]
            losses["graph"] = self.graph_loss_fn(predicted_edges, true_edges)
        
        # Intervention prediction loss (simplified)
        if "intervention_representations" in outputs:
            # Self-reconstruction loss as proxy
            hidden = outputs.get("hidden_states", outputs["intervention_representations"])
            if len(hidden.shape) > 2:
                hidden = hidden.mean(dim=1)
            recon_loss = F.mse_loss(
                outputs["intervention_representations"], 
                hidden.detach()
            )
            losses["intervention"] = recon_loss
        
        return losses
    
    def save_model(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config
        }
        torch.save(checkpoint, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.logger.info(f"Model loaded from {path}")


class CausalFoundationModelHub:
    """Hub for managing pre-trained causal foundation models."""
    
    def __init__(self, model_dir: str = "causal_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.logger = get_logger("causalllm.causal_foundation_model_hub")
        self.available_models = self._scan_available_models()
    
    def _scan_available_models(self) -> Dict[str, Dict[str, Any]]:
        """Scan for available pre-trained models."""
        models = {}
        
        for model_file in self.model_dir.glob("*.pt"):
            try:
                # Load model metadata
                metadata_file = model_file.with_suffix(".json")
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    models[model_file.stem] = metadata
            except Exception as e:
                self.logger.warning(f"Could not load metadata for {model_file}: {e}")
        
        return models
    
    def load_pretrained_model(self, model_name: str, 
                            device: str = "cpu") -> CausalFoundationModel:
        """Load a pre-trained causal foundation model."""
        
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not found. Available: {list(self.available_models.keys())}")
        
        model_path = self.model_dir / f"{model_name}.pt"
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
            config = checkpoint["config"]
            
            # Determine vocab size (simplified)
            vocab_size = 10000  # Would be stored in metadata
            
            model = CausalFoundationModel(config, vocab_size)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(device)
            model.eval()
            
            self.logger.info(f"Loaded pre-trained model: {model_name}")
            return model
            
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def save_model(self, model: CausalFoundationModel, 
                  name: str, metadata: Dict[str, Any]):
        """Save a model to the hub."""
        model_path = self.model_dir / f"{name}.pt"
        metadata_path = self.model_dir / f"{name}.json"
        
        # Save model
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": model.config
        }
        torch.save(checkpoint, model_path)
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.available_models[name] = metadata
        self.logger.info(f"Model {name} saved to hub")
    
    def list_models(self) -> List[str]:
        """List available models."""
        return list(self.available_models.keys())


class CausalRepresentationLearner:
    """Learn causal representations from observational data."""
    
    def __init__(self, foundation_model: CausalFoundationModel):
        self.foundation_model = foundation_model
        self.logger = get_logger("causalllm.causal_representation_learner")
    
    def learn_representations(self, datasets: List[pd.DataFrame],
                            variable_descriptions: List[Dict[str, str]],
                            domains: List[str]) -> List[CausalRepresentation]:
        """Learn causal representations from multiple datasets."""
        
        representations = []
        
        for i, (data, variables, domain) in enumerate(zip(datasets, variable_descriptions, domains)):
            self.logger.info(f"Learning representations for dataset {i+1}/{len(datasets)}")
            
            # Convert data to causal training examples (simplified)
            examples = self._convert_data_to_examples(data, variables, domain)
            
            # Create dataset
            dataset = CausalDataset(examples, max_variables=self.foundation_model.config.max_variables)
            
            # Extract representations using foundation model
            all_reps = []
            for example in examples:
                rep = self.foundation_model.encode_causal_system(
                    example.variables, example.graph_edges, domain
                )
                all_reps.append(rep)
            
            # Aggregate representations
            if all_reps:
                aggregated_rep = self._aggregate_representations(all_reps)
                representations.append(aggregated_rep)
        
        return representations
    
    def _convert_data_to_examples(self, data: pd.DataFrame, 
                                 variables: Dict[str, str],
                                 domain: str) -> List[CausalTrainingExample]:
        """Convert raw data to training examples (simplified)."""
        
        # For now, create a simple example with all variables
        # In practice, this would involve more sophisticated processing
        
        variable_dict = {}
        for col in data.columns:
            if col in variables:
                variable_dict[col] = variables[col]
        
        # Create simple fully connected graph as placeholder
        graph_edges = []
        vars_list = list(variable_dict.keys())
        for i, var1 in enumerate(vars_list):
            for j, var2 in enumerate(vars_list):
                if i != j and np.random.random() < 0.3:  # Random sparse edges
                    graph_edges.append((var1, var2))
        
        example = CausalTrainingExample(
            variables=variable_dict,
            graph_edges=graph_edges,
            interventions={},
            outcomes={},
            counterfactuals={},
            domain=domain,
            context=f"Dataset with {len(data)} samples"
        )
        
        return [example]
    
    def _aggregate_representations(self, representations: List[CausalRepresentation]) -> CausalRepresentation:
        """Aggregate multiple representations into one."""
        
        if not representations:
            raise ValueError("No representations to aggregate")
        
        if len(representations) == 1:
            return representations[0]
        
        # Average embeddings
        var_embeddings = torch.stack([rep.variable_embeddings for rep in representations]).mean(dim=0)
        graph_embeddings = torch.stack([rep.graph_embedding for rep in representations]).mean(dim=0)
        
        # Aggregate mechanisms
        all_mechanisms = {}
        for rep in representations:
            for key, mechanism in rep.causal_mechanisms.items():
                if key not in all_mechanisms:
                    all_mechanisms[key] = []
                all_mechanisms[key].append(mechanism)
        
        aggregated_mechanisms = {}
        for key, mechanisms in all_mechanisms.items():
            aggregated_mechanisms[key] = torch.stack(mechanisms).mean(dim=0)
        
        return CausalRepresentation(
            variable_embeddings=var_embeddings,
            graph_embedding=graph_embeddings,
            causal_mechanisms=aggregated_mechanisms,
            intervention_embeddings={},  # Would aggregate similarly
            metadata={"aggregated": True, "num_sources": len(representations)}
        )


# Convenience functions
def create_causal_foundation_model(model_type: CausalModelType = CausalModelType.UNIFIED_CAUSAL,
                                 architecture: CausalModelArchitecture = CausalModelArchitecture.TRANSFORMER,
                                 vocab_size: int = 10000,
                                 **kwargs) -> CausalFoundationModel:
    """Create a causal foundation model with specified configuration."""
    
    config = CausalModelConfig(
        model_type=model_type,
        architecture=architecture,
        **kwargs
    )
    
    return CausalFoundationModel(config, vocab_size)


def load_pretrained_causal_model(model_name: str, 
                               model_hub: Optional[CausalFoundationModelHub] = None) -> CausalFoundationModel:
    """Load a pre-trained causal foundation model."""
    
    if model_hub is None:
        model_hub = CausalFoundationModelHub()
    
    return model_hub.load_pretrained_model(model_name)


async def train_causal_foundation_model(training_data: List[CausalTrainingExample],
                                      model_config: CausalModelConfig,
                                      num_epochs: int = 10,
                                      batch_size: int = 32,
                                      device: str = "cpu") -> CausalFoundationModel:
    """Train a causal foundation model from scratch."""
    
    logger = get_logger("causalllm.train_causal_foundation_model")
    
    # Create dataset
    dataset = CausalDataset(training_data, max_variables=model_config.max_variables)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    vocab_size = len(dataset.variable_vocab)
    model = CausalFoundationModel(model_config, vocab_size)
    
    # Create trainer
    trainer = CausalFoundationModelTrainer(model, model_config, device)
    
    logger.info(f"Starting training for {num_epochs} epochs...")
    
    # Training loop
    for epoch in range(num_epochs):
        losses = trainer.train_epoch(dataloader)
        logger.info(f"Epoch {epoch+1}/{num_epochs}: Total Loss = {losses['total']:.4f}")
    
    logger.info("Training completed")
    return model