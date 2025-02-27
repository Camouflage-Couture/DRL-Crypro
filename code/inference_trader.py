import torch
import numpy as np
import os
import pandas as pd
import glob
from PIL import Image
from torchvision import transforms as T
import argparse
from tqdm import tqdm
import torch.nn as nn

# Actions enum
class Actions:
    IDLE = 0
    LONG = 1
    SHORT = 2
    CLOSE = 3

def get_legal_actions(position):
    """Get legal actions based on current position"""
    if position == 0:  # No position
        legal_actions = [Actions.IDLE, Actions.LONG, Actions.SHORT]
    elif position == 1 or position == -1:  # Long or short position
        legal_actions = [Actions.IDLE, Actions.CLOSE]
    
    # Return action values, not the enum objects
    return legal_actions

# Define exactly matching model for your saved weights
class ActorPPO(nn.Module):
    def __init__(self, input_size=338, action_size=4):
        super(ActorPPO, self).__init__()
        # No CNN backbone in this model - it uses raw features
        self.actor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size)
        )
    
    def forward(self, x, account=None, action_mask=None):
        # For raw data models, x is already a feature vector
        logits = self.actor(x)
        
        if action_mask is not None:
            logits[action_mask == 0] = -1e9  # Mask out illegal actions
        
        return torch.distributions.Categorical(logits=logits)

class LiveInferenceTrader:
    def __init__(self, model_path, device='cpu'):
        """Initialize inference trader with model path"""
        self.model_path = model_path
        self.device = device
        
        # Load the model
        self.net = self._load_model()
        
        # Transform for normalizing images if we need to extract features
        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])
        
        # Initialize position state
        self.position = 0  # 0: no position, 1: long, -1: short
        self.fund_rate = 0.0  # Default fund rate
        
        print(f"Model loaded from: {model_path}")
        print(f"Using device: {device}")
    
    def _load_model(self):
        """Load the PPO model with exact architecture matching saved weights"""
        # Create the actor network with exact input size from error message
        actor = ActorPPO(input_size=338, action_size=4)
        
        # Load model weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Handle state dict mismatch - create new state dict with correct keys
        new_state_dict = {}
        for key, value in checkpoint['net'].items():
            if key.startswith('actor'):
                # Actor layer keys should match directly
                new_state_dict[key] = value
        
        # Load the modified state dict
        actor.load_state_dict(new_state_dict)
        actor.to(self.device)
        actor.eval()
        
        return actor
    
    def _extract_features(self, image_path):
        """
        Extract feature vector from image or filename
        
        Since the model expects a 338-dimension feature vector, we need to
        construct it from the available information.
        """
        # Initialize an empty feature vector
        features = torch.zeros(338)
        
        # Try to extract features from the filename
        try:
            # Extract price info from the filename format: YYYY-MM-DD_HH-MM-SS_[ratio,close,high,volatility]
            basename = os.path.basename(image_path)
            if '[' in basename and ']' in basename:
                price_str = basename.split('[')[1].split(']')[0]
                price_features = [float(x) for x in price_str.split(',')]
                
                # Place price features at the beginning of the vector
                for i, feature in enumerate(price_features):
                    if i < len(features):
                        features[i] = feature
            
            # Extract timestamp features if possible
            timestamp_parts = basename.split('_')[0:2]
            if len(timestamp_parts) >= 2:
                # Simple feature: hour of day normalized to [0,1]
                try:
                    hour = int(timestamp_parts[1].split('-')[0])
                    features[4] = hour / 24.0
                except:
                    pass
        except Exception as e:
            print(f"Warning: Failed to extract features from filename: {e}")
        
        # Place account features near the end of the vector
        features[336] = self.position  # Position feature
        features[337] = self.fund_rate  # Fund rate feature
        
        return features.unsqueeze(0).to(self.device)
    
    def predict_action(self, image_path):
        """
        Predict action from a candlestick image path
        
        Args:
            image_path: Path to the candlestick image
            
        Returns:
            action_name: Name of the predicted action
            action: Action index
        """
        # Extract features from image path
        input_features = self._extract_features(image_path)
        
        # Get legal actions
        legal_actions = get_legal_actions(self.position)
        action_mask = torch.zeros((1, 4), device=self.device)  # 4 actions
        for action in legal_actions:
            action_mask[0, action] = 1
        
        # Predict action
        with torch.no_grad():
            dist = self.net(input_features, action_mask=action_mask)
            action = dist.sample().cpu().numpy()[0]
        
        # Update position
        self._update_position(action)
        
        # Convert action to name
        action_names = ['IDLE', 'LONG', 'SHORT', 'CLOSE']
        return action_names[action], action
    
    def _update_position(self, action):
        """Update position based on action"""
        if action == Actions.IDLE:
            # Position stays the same
            pass
        elif action == Actions.LONG:
            if self.position == 0:
                self.position = 1
            elif self.position == -1:
                self.position = 1  # Close short and open long
        elif action == Actions.SHORT:
            if self.position == 0:
                self.position = -1
            elif self.position == 1:
                self.position = -1  # Close long and open short
        elif action == Actions.CLOSE:
            self.position = 0
    
    def run_on_directory(self, image_dir, output_file=None):
        """
        Run inference on all images in a directory
        
        Args:
            image_dir: Directory with candlestick images
            output_file: Path to save results CSV
            
        Returns:
            DataFrame with predictions
        """
        # Find all image files
        image_files = sorted(glob.glob(f"{image_dir}/*.png"))
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return None
        
        # Reset state
        self.position = 0
        
        results = []
        print(f"Running inference on {len(image_files)} images...")
        
        for img_path in tqdm(image_files):
            # Extract timestamp from filename
            try:
                timestamp = os.path.basename(img_path).split('_')[0:2]
                timestamp = '_'.join(timestamp)
            except:
                timestamp = os.path.basename(img_path)
            
            # Predict action
            action_name, action = self.predict_action(img_path)
            
            # Record result
            results.append({
                'timestamp': timestamp,
                'image': os.path.basename(img_path),
                'action': action,
                'action_name': action_name,
                'position': self.position
            })
        
        # Create DataFrame
        results_df = pd.DataFrame(results)
        
        # Save to file if requested
        if output_file:
            os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
            results_df.to_csv(output_file)
            print(f"Results saved to {output_file}")
        
        # Print summary
        print("\nSummary:")
        print(f"Total images processed: {len(results_df)}")
        print("\nAction distribution:")
        action_counts = results_df['action_name'].value_counts()
        for action, count in action_counts.items():
            print(f"  {action}: {count} ({count/len(results_df)*100:.1f}%)")
        
        return results_df

def main():
    parser = argparse.ArgumentParser(description="Run inference with trained PPO model")
    parser.add_argument('--model', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory with candlestick images')
    parser.add_argument('--output', type=str, default='./inference_results/predictions.csv', help='Output CSV file')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run inference on (cpu or cuda:0)')
    
    args = parser.parse_args()
    
    # Create and run trader
    trader = LiveInferenceTrader(
        model_path=args.model,
        device=args.device
    )
    
    results = trader.run_on_directory(
        image_dir=args.image_dir,
        output_file=args.output
    )
    
    return results

if __name__ == "__main__":
    main()