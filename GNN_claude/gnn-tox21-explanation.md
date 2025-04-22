# Graph Neural Network Implementation for Tox21 Dataset

This document explains the implementation of Graph Neural Networks (GNNs) for predicting molecular toxicity using the Tox21 dataset. The code addresses Part 2 of the assignment: implementing GNN models to predict activity against 12 different toxicity-related biological targets.

## 1. Overview of the Tox21 Dataset

The Tox21 dataset contains approximately 8,000 molecules with binary classification labels for 12 different toxicity targets. From our initial analysis:

- The dataset has 8,006 molecules (rows)
- Each molecule has activity labels for 12 toxicity targets (some values are missing)
- All molecules have SMILES string representations
- The targets are highly imbalanced, with most molecules being inactive (negative class)
- There are significant amounts of missing data for some targets

The implementation includes:
- Converting SMILES strings to molecular graphs using RDKit
- Feature extraction for atoms and bonds
- Creating PyTorch Geometric Data objects to represent molecules
- Implementing two GNN architectures:
  - Graph Convolutional Network (GCN)
  - Message Passing Neural Network (MPNN)
- Training and evaluating the models on the Tox21 dataset

## 2. Data Processing

### 2.1. Feature Extraction

Before converting SMILES to graphs, we need to define the features for atoms and bonds:

```python
def get_atom_features(atom):
    """
    Returns an array of atom features.
    """
    # Atom type (one-hot encoded)
    atom_type_one_hot = np.zeros(100)
    atom_num = atom.GetAtomicNum()
    if atom_num < 100:
        atom_type_one_hot[atom_num] = 1
    
    # Other atom features
    formal_charge = atom.GetFormalCharge()
    hybridization = atom.GetHybridization()
    is_aromatic = int(atom.GetIsAromatic())
    num_h = atom.GetTotalNumHs()
    
    # Hybridization (one-hot encoded)
    hybridization_one_hot = np.zeros(6)
    if hybridization.name in ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2']:
        hyb_idx = ['S', 'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'].index(hybridization.name)
        hybridization_one_hot[hyb_idx] = 1
    
    # Combine all features
    atom_features = np.concatenate([
        atom_type_one_hot,
        np.array([formal_charge + 4]),  # Shift +4 to ensure positive values
        hybridization_one_hot,
        np.array([is_aromatic]),
        np.array([num_h])
    ])
    
    return atom_features

def get_bond_features(bond):
    """
    Returns an array of bond features.
    """
    # Bond type (one-hot encoded)
    bond_type_one_hot = np.zeros(4)
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        bond_type_one_hot[0] = 1
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        bond_type_one_hot[1] = 1
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        bond_type_one_hot[2] = 1
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        bond_type_one_hot[3] = 1
    
    # Other bond features
    is_conjugated = int(bond.GetIsConjugated())
    is_in_ring = int(bond.IsInRing())
    
    # Combine all features
    bond_features = np.concatenate([
        bond_type_one_hot,
        np.array([is_conjugated, is_in_ring])
    ])
    
    return bond_features
```

These functions extract meaningful chemical features:

For atoms:
- Atomic number (one-hot encoded)
- Formal charge
- Hybridization type
- Aromaticity
- Number of hydrogen atoms

For bonds:
- Bond type (single, double, triple, aromatic)
- Conjugation
- Presence in ring

### 2.2. SMILES to Molecular Graphs

The first step is to convert SMILES strings into molecular graphs:

```python
def smiles_to_graph(smiles):
    """
    Converts a SMILES string to a PyTorch Geometric Data object containing the molecular graph.
    """
    # Convert SMILES to RDKit molecule
    mol = Chem.MolFromSmiles(smiles)
    
    # Extract atom features
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(get_atom_features(atom))
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float)
    
    # Extract bond features and create edge indices
    edge_indices = []
    edge_features_list = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        
        edge_indices.append([i, j])
        edge_indices.append([j, i])  # Add reverse edge for undirected graph
        
        edge_features = get_bond_features(bond)
        edge_features_list.append(edge_features)
        edge_features_list.append(edge_features)  # Duplicate for reverse edge
    
    # Create PyTorch Geometric Data object
    edge_index = torch.tensor(np.array(edge_indices).T, dtype=torch.long)
    edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.float)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    return data
```

This function performs several important steps:
1. Converts the SMILES string to an RDKit molecule object
2. Extracts atom features for each atom in the molecule
3. Creates edges between connected atoms (both directions for undirected graph)
4. Extracts bond features for each bond
5. Builds a PyTorch Geometric `Data` object with node features, edge indices, and edge features

### 2.3. Creating the Dataset

We create a custom PyTorch Dataset to handle the Tox21 data:

```python
class Tox21Dataset(Dataset):
    def __init__(self, dataframe, target_columns, smiles_column='smiles'):
        self.dataframe = dataframe.reset_index(drop=True)
        self.target_columns = target_columns
        self.smiles_column = smiles_column
        
        # Convert SMILES to molecular graphs
        self.graphs = []
        self.valid_indices = []
        
        for idx, row in tqdm(self.dataframe.iterrows(), total=len(self.dataframe), desc="Converting SMILES to graphs"):
            graph = smiles_to_graph(row[self.smiles_column])
            if graph is not None:
                self.graphs.append(graph)
                self.valid_indices.append(idx)
        
        # Only keep rows with valid graphs
        self.dataframe = self.dataframe.iloc[self.valid_indices].reset_index(drop=True)
    
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        graph = self.graphs[idx]
        
        # Get targets (handling NaN values)
        targets = []
        for col in self.target_columns:
            value = self.dataframe.loc[idx, col]
            # Convert NaN to -1 (will be masked during loss calculation)
            targets.append(-1 if pd.isna(value) else value)
            
        return graph, torch.tensor(targets, dtype=torch.float)
```

This class:
1. Converts all valid SMILES strings to molecular graphs
2. Filters out invalid molecules
3. Handles missing values in the target columns by setting them to -1 (to be masked during training)
4. Returns graph-target pairs for training

## 3. Model Architectures

Two GNN architectures are implemented to compare their performance:

### 3.1. Graph Convolutional Network (GCN)

```python
class GCNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, output_dim):
        super(GCNModel, self).__init__()
        
        # GCN layers
        self.conv1 = GCNConv(node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Apply GCN layers with residual connections
        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index)) + x1
        x3 = F.relu(self.conv3(x2, edge_index)) + x2
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x3, batch)
        
        # Apply MLP for final prediction
        out = self.mlp(x)
        
        return out
```

The GCN model:
1. Uses PyTorch Geometric's `GCNConv` layers to update node representations based on neighboring nodes
2. Incorporates residual connections to help with gradient flow during training
3. Applies global mean pooling to convert node-level features to a graph-level representation
4. Uses a multi-layer perceptron (MLP) to make the final predictions for the 12 targets

### 3.2. Message Passing Neural Network (MPNN)

```python
class MPNNLayer(MessagePassing):
    def __init__(self, node_dim, edge_dim, hidden_dim):
        super(MPNNLayer, self).__init__(aggr='add')
        
        # MLPs for message passing
        self.message_mlp = nn.Sequential(
            nn.Linear(node_dim + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Update function
        self.update_mlp = nn.Sequential(
            nn.Linear(node_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
    def forward(self, x, edge_index, edge_attr):
        # Start propagating messages
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_i, x_j, edge_attr):
        # Create messages based on source nodes and edge features
        message_input = torch.cat([x_j, edge_attr], dim=1)
        return self.message_mlp(message_input)
    
    def update(self, aggr_out, x):
        # Update node embeddings
        update_input = torch.cat([x, aggr_out], dim=1)
        return self.update_mlp(update_input)

class MPNNModel(nn.Module):
    def __init__(self, node_features, edge_features, hidden_dim, output_dim):
        super(MPNNModel, self).__init__()
        
        # MPNN layers
        self.mpnn1 = MPNNLayer(node_features, edge_features, hidden_dim)
        self.mpnn2 = MPNNLayer(hidden_dim, edge_features, hidden_dim)
        self.mpnn3 = MPNNLayer(hidden_dim, edge_features, hidden_dim)
        
        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        
        # Apply MPNN layers with residual connections
        x1 = F.relu(self.mpnn1(x, edge_index, edge_attr))
        x2 = F.relu(self.mpnn2(x1, edge_index, edge_attr)) + x1
        x3 = F.relu(self.mpnn3(x2, edge_index, edge_attr)) + x2
        
        # Global pooling to get graph-level representation
        x = global_mean_pool(x3, batch)
        
        # Apply MLP for final prediction
        out = self.mlp(x)
        
        return out
```

The MPNN model:
1. Implements a custom `MessagePassing` layer that explicitly models:
   - Message creation (combining source node and edge features)
   - Message aggregation (summing messages from neighbors)
   - Node update (combining current node state with aggregated messages)
2. Stacks multiple MPNN layers with residual connections
3. Also uses global mean pooling and an MLP for final predictions

The key difference from GCN is that MPNN explicitly uses edge features during message passing, potentially capturing more detailed molecular information.

## 4. Training and Evaluation

### 4.1. Loss Function

A key challenge in the Tox21 dataset is handling missing values. We implement a masked BCE loss:

```python
def masked_bce_loss(pred, target):
    # Create a mask for non-missing values (where target != -1)
    mask = (target != -1).float()
    
    # Apply sigmoid to get probabilities
    pred_probs = torch.sigmoid(pred)
    
    # Compute BCE loss only for non-missing values
    loss = F.binary_cross_entropy_with_logits(pred, target * mask, reduction='none')
    loss = loss * mask  # Zero out loss for missing values
    
    # Compute mean loss over non-missing values
    non_missing = mask.sum()
    if non_missing > 0:
        return loss.sum() / non_missing
    else:
        return torch.tensor(0.0, device=pred.device)
```

This function:
1. Creates a mask to identify non-missing values
2. Computes BCE loss only for non-missing values
3. Normalizes the loss by the number of non-missing values

### 4.2. Training Loop

The training function handles both training and validation:

```python
def train_model(model, train_loader, val_loader, optimizer, num_epochs=50, device='cuda'):
    model.to(device)
    best_val_roc_auc = 0
    best_model_state = None
    
    train_losses = []
    val_losses = []
    val_roc_aucs = []
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        epoch_loss = 0
        
        for data, targets in train_loader:
            data = data.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(data)
            loss = masked_bce_loss(outputs, targets)
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(targets)
        
        avg_train_loss = epoch_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for data, targets in val_loader:
                data = data.to(device)
                targets = targets.to(device)
                
                # Forward pass
                outputs = model(data)
                loss = masked_bce_loss(outputs, targets)
                
                val_loss += loss.item() * len(targets)
                
                # Store predictions and targets for ROC-AUC calculation
                all_targets.append(targets.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())
        
        avg_val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
        
        # Compute ROC-AUC for each task
        all_targets = np.vstack(all_targets)
        all_outputs = np.vstack(all_outputs)
        
        task_roc_aucs = []
        for task_idx in range(len(target_columns)):
            # Get mask for non-missing values
            mask = (all_targets[:, task_idx] != -1)
            if mask.sum() > 0 and len(np.unique(all_targets[mask, task_idx])) > 1:
                y_true = all_targets[mask, task_idx]
                y_score = all_outputs[mask, task_idx]
                try:
                    roc_auc = roc_auc_score(y_true, y_score)
                    task_roc_aucs.append(roc_auc)
                except:
                    pass
        
        if task_roc_aucs:
            mean_roc_auc = np.mean(task_roc_aucs)
            val_roc_aucs.append(mean_roc_auc)
            
            # Save best model
            if mean_roc_auc > best_val_roc_auc:
                best_val_roc_auc = mean_roc_auc
                best_model_state = model.state_dict().copy()
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses, val_roc_aucs
```

Key aspects of the training process:
1. Uses ROC-AUC as the primary validation metric (appropriate for imbalanced classification)
2. Handles missing values in both loss calculation and metric computation
3. Saves the best model based on validation ROC-AUC
4. Returns training history for later visualization

### 4.3. Evaluation

The evaluation function computes metrics for each target separately:

```python
def evaluate_model(model, test_loader, device='cuda'):
    model.eval()
    all_targets = []
    all_outputs = []
    
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            
            # Forward pass
            outputs = model(data)
            
            # Store predictions and targets
            all_targets.append(targets.cpu().numpy())
            all_outputs.append(outputs.cpu().numpy())
    
    all_targets = np.vstack(all_targets)
    all_outputs = np.vstack(all_outputs)
    
    # Compute metrics for each task
    task_metrics = {}
    for task_idx, task_name in enumerate(target_columns):
        # Get mask for non-missing values
        mask = (all_targets[:, task_idx] != -1)
        if mask.sum() > 0 and len(np.unique(all_targets[mask, task_idx])) > 1:
            y_true = all_targets[mask, task_idx]
            y_score = all_outputs[mask, task_idx]
            y_pred = (y_score > 0).astype(int)  # Threshold at 0 (sigmoid=0.5)
            
            try:
                task_roc_auc = roc_auc_score(y_true, y_score)
                task_accuracy = accuracy_score(y_true, y_pred)
                
                task_metrics[task_name] = {
                    'roc_auc': task_roc_auc,
                    'accuracy': task_accuracy
                }
            except:
                task_metrics[task_name] = {
                    'roc_auc': np.nan,
                    'accuracy': np.nan
                }
    
    return task_metrics
```

This function:
1. Computes ROC-AUC and accuracy for each toxicity target
2. Handles missing values and potential errors (e.g., when a target has only one class in the test set)
3. Returns detailed metrics for further analysis and comparison

## 5. Main Execution Flow

The `main()` function orchestrates the entire workflow:

1. **Data Preparation**:
   - Load the Tox21 CSV file
   - Split into train/validation/test sets (64%/16%/20%)
   - Create PyTorch Datasets and DataLoaders

2. **Model Training**:
   - Initialize both GCN and MPNN models
   - Train each model for 30 epochs
   - Track training/validation losses and ROC-AUC scores

3. **Evaluation**:
   - Evaluate both models on the test set
   - Compute and display metrics for each toxicity target
   - Compare overall performance between models

4. **Visualization**:
   - Plot training curves (loss and ROC-AUC)
   - Create a bar chart comparing model performance across targets

## 6. Expected Outcomes

Based on the implementation, we expect:

1. Both GNN architectures should outperform traditional ML baselines (from Part 1) for most targets
2. The MPNN model might perform better than GCN since it explicitly uses edge features
3. Performance will vary across targets due to differences in:
   - Class imbalance
   - Amount of missing data
   - Inherent difficulty of predicting each toxicity endpoint

This GNN-based approach demonstrates how molecular graphs can be used directly for property prediction, leveraging the structural information of molecules rather than relying on engineered fingerprints.