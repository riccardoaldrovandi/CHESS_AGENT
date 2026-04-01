# model.py
import torch
import torch.nn as nn

class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # --- CONVOLUTIONAL BACKBONE (The "Visual" Feature Extractor) ---
        # Analyzes the spatial arrangement of pieces on the 8x8 board.
        self.conv_block = nn.Sequential(
            # First layer: takes 12 input planes (6 pieces x 2 colors) and creates 64 feature maps.
            # kernel_size=3 looks at 3x3 grids. padding=1 maintains the 8x8 spatial dimension.
            nn.Conv2d(12, 64, kernel_size=3, padding=1),
            nn.ReLU(), # Activation function: allows the network to learn non-linear patterns
            
            # Second layer: deepens the analysis by increasing feature maps from 64 to 128.
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU()
        )
        
        # --- FLATTENING LAYER ---
        # Transforms the 128x8x8 data cube into a single flat vector.
        # 128 * 8 * 8 = 8192 elements.
        self.flatten = nn.Flatten()
        
        # --- POLICY HEAD (The "Move Suggester") ---
        # Determines the probability distribution over all possible moves.
        self.policy_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512), # Fully connected layer
            nn.ReLU(),
            nn.Linear(512, 4096) # Outputting logits for approx. 4096 possible move combinations
        )
        
        # --- VALUE HEAD (The "Position Judge") ---
        # Evaluates the current board state's winning probability.
        self.value_head = nn.Sequential(
            nn.Linear(128 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1), # Single scalar output
            nn.Tanh() # Compresses the result between -1 (Loss) and 1 (Win)
        )

    def forward(self, x):
        """
        Defines the data flow (Forward Pass):
        x -> Convolutional Block -> Flattening -> Policy & Value Heads
        """
        x = self.conv_block(x)    # Extract spatial features
        x = self.flatten(x)       # Prepare data for dense layers
        
        policy = self.policy_head(x) # Compute move probabilities (logits)
        value = self.value_head(x)   # Compute position evaluation
        
        return policy, value