import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
import warnings

warnings.filterwarnings("ignore")

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=6, num_layers=4):
        super().__init__()
        # Embedding layer to convert token IDs to vectors
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding to give the model information about token positions
        self.pos_encoder = nn.Embedding(1024, d_model)
        
        # Create a single transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,  # Dimension of the model
            nhead=nhead,      # Number of attention heads
            dim_feedforward=4*d_model,  # Dimension of the feedforward network
            batch_first=True  # Input is (batch, seq, feature)
        )
        
        # Create the full transformer encoder with multiple layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Final linear layer to convert back to vocabulary size
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        # Generate position indices
        pos = torch.arange(0, src.size(1)).unsqueeze(0).repeat(src.size(0), 1).to(src.device)
        
        # Combine token embeddings and positional encodings
        src_embed = self.embedding(src) + self.pos_encoder(pos)
        
        # Generate attention mask to prevent looking at future tokens
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        # Pass through the transformer
        out = self.transformer(src_embed, src_mask)
        
        # Project to vocabulary size
        return self.fc_out(out)

    def generate_square_subsequent_mask(self, sz):
        # Generate a square mask for the sequence. The mask ensures that each
        # position can only attend to earlier positions in the sequence.
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        
        # Tokenize each text and store the resulting input IDs and attention masks
        for text in texts:
            encodings = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings['input_ids']))
            self.attn_masks.append(torch.tensor(encodings['attention_mask']))

    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.input_ids)

    def __getitem__(self, idx):
        # Return the input IDs and attention mask for a given index
        return self.input_ids[idx], self.attn_masks[idx]

def train(model, train_loader, optimizer, scheduler, criterion, device, scaler):
    model.train()  # Set the model to training mode
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        
        optimizer.zero_grad()  # Reset gradients
        
        with torch.cuda.amp.autocast():
            outputs = model(input_ids)
            # Calculate loss
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
        
        # Perform backpropagation with mixed precision
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)  # Return average loss

def evaluate(model, val_loader, criterion, device):
    model.eval()  # Set the model to evaluation mode
    total_loss = 0
    with torch.no_grad():  # Disable gradient calculation
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)  # Return average loss

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_k=50, device='cuda'):
    model.eval()  # Set the model to evaluation mode
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():  # Disable gradient calculation
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature  # Apply temperature
            # Get top k logits and indices
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)  # Convert to probabilities
            next_token = torch.multinomial(probs, num_samples=1)  # Sample next token
            next_token = top_k_indices.gather(-1, next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)  # Add new token to input
            
            if next_token.item() == tokenizer.eos_token_id:
                break  # Stop if end of sequence token is generated
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    # Determine if CUDA (GPU) is available, otherwise use CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # If using CUDA, print GPU information
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")  # Get the name of the first GPU
        print(f"CUDA Version: {torch.version.cuda}")  # Print CUDA version

    # Set hyperparameters
    BATCH_SIZE = 8  # Number of samples processed before the model is updated
    EPOCHS = 20  # Number of complete passes through the training dataset
    LEARNING_RATE = 3e-4  # Step size at each iteration while moving toward a minimum of the loss function
    WARMUP_STEPS = 100  # Number of steps for the warmup phase of learning rate scheduler

    # Initialize the tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')  # Load pre-trained GPT-2 tokenizer
    tokenizer.pad_token = tokenizer.eos_token  # Set padding token to be the same as the end of sequence token
    vocab_size = tokenizer.vocab_size  # Get the size of the vocabulary for the model

    # Texts used for a basic training purpose, u can add remove, just grant ur functionality and words
    texts = [
    "Artificial intelligence is revolutionizing industries across the globe.",
    "Machine learning algorithms can identify patterns in vast amounts of data.",
    "Neural networks are inspired by the structure and function of the human brain.",
    "Deep learning has achieved remarkable results in image and speech recognition.",
    "AI-powered chatbots are transforming customer service interactions.",
    "Reinforcement learning allows AI agents to learn through trial and error.",
    "Computer vision systems can analyze and interpret visual information from the world.",
    "Natural language processing enables machines to understand and generate human language.",
    "AI is playing a crucial role in the development of autonomous vehicles.",
    "Ethical considerations are paramount in the development and deployment of AI systems.",
    "AI algorithms are being used to predict and prevent cyber attacks.",
    "Generative AI models can create new content, including text, images, and music.",
    "AI-driven robotics is transforming manufacturing and automation processes.",
    "Machine learning models require large datasets for training and validation.",
    "Transfer learning allows AI models to apply knowledge from one task to another.",
    "Explainable AI aims to make machine learning models more interpretable and transparent.",
    "AI is being used to accelerate drug discovery and development in pharmaceuticals.",
    "Federated learning enables AI models to be trained across decentralized devices.",
    "AI-powered recommendation systems drive personalized experiences in e-commerce and streaming services.",
    "The field of AI ethics addresses concerns about bias, privacy, and accountability.",
    "Quantum computing has the potential to significantly advance AI capabilities.",
    "AI is being applied in climate modeling to better predict and mitigate climate change.",
    "Facial recognition technology, powered by AI, raises both security benefits and privacy concerns.",
    "AI-driven predictive maintenance is optimizing industrial operations and reducing downtime.",
    "Natural language generation systems can produce human-like text for various applications.",
    "AI is enhanacing medical imaging analysis, improving disease detection and diagnosis.",
    "Robotic process automation is streamlining repetitive tasks in business operations.",
    "AI-powered virtual assistants are becoming increasingly sophisticated and widespread.",
    "Machine learning is being used to detect financial fraud and anomalies in real-time.",
    "AI algorithms are optimizing energy consumption in smart buildings and cities.",
    "Sentiment analysis powered by AI is helping businesses understand customer opinions at scale.",
    "AI is being used in agriculture for crop monitoring, yield prediction, and precision farming.",
    "The development of artificial general intelligence remains a long-term goal in AI research.",
    "AI-driven simulations are accelerating scientific research across various disciplines.",
    "Edge AI brings machine learning capabilities to IoT devices and local hardware.",
    "AI is enhancing cybersecurity through anomaly detection and threat intelligence.",
    "Conversational AI is improving human-computer interactions in various applications.",
    "AI-powered tools are assisting in content creation, from writing to video editing.",
    "Machine learning models can suffer from biases present in their training data.",
    "AI is being used to optimize supply chain management and logistics operations.",
    "Neuromorphic computing aims to create AI hardware that mimics the human brain's architecture.",
    "AI-driven personalized learning systems are tailoring education to individual student needs.",
    "Generative adversarial networks (GANs) are pushing the boundaries of AI-created content.",
    "AI is being applied in weather forecasting to improve accuracy and extend prediction timeframes.",
    "Automated machine learning (AutoML) is making AI more accessible to non-experts.",
    "AI-powered predictive policing raises both potential benefits and ethical concerns.",
    "Swarm intelligence algorithms draw inspiration from collective behavior in nature.",
    "AI is enhancing game development, creating more realistic and adaptive gaming experiences.",
    "The interpretability of AI decision-making processes is crucial for building trust in AI systems.",
    "AI-driven analytics are providing valuable insights for business strategy and decision-making."
]

    # Create a dataset from the texts using the custom TextDataset class
    full_dataset = TextDataset(texts, tokenizer)
    
    # Split the dataset into training (80%) and validation (20%) sets
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    # Create DataLoaders for efficient batching and parallel data loading
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=2)

    # Initialize the model and move it to the specified device (CPU or GPU)
    model = TransformerModel(vocab_size).to(device)

    # Define the loss function (criterion)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    # Initialize the optimizer with weight decay for regularization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Calculate total training steps and create a learning rate scheduler
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    # Initialize the gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Create a directory to save model checkpoints
    os.makedirs("checkpoints", exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        # Train the model for one epoch and get the average training loss
        train_loss = train(model, train_loader, optimizer, scheduler, criterion, device, scaler)
        
        # Evaluate the model on the validation set and get the average validation loss
        val_loss = evaluate(model, val_loader, criterion, device)
        
        # Print the training progress
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save a checkpoint of the model
        checkpoint_path = f"checkpoints/model_checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        # If this is the best model so far (lowest validation loss), remember it
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoint_path
            print(f"New best model saved: {best_checkpoint_path}")
        
        # Early stopping: If validation loss hasn't improved for more than 10 epochs, stop training
        if epoch > 10 and val_loss > best_val_loss:
            print("Early stopping")
            break

    print("Training completed!")
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    prompt = "Artificial intelligence is"
    generated_text = generate_text(model, tokenizer, prompt, device=device)
    print(f"Generated text: {generated_text}")

if __name__ == "__main__":
    main()