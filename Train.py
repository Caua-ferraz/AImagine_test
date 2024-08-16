import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
import warnings
from ai_datasets import get_ai_texts

warnings.filterwarnings("ignore")

class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=6, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(1024, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src):
        pos = torch.arange(0, src.size(1)).unsqueeze(0).repeat(src.size(0), 1).to(src.device)
        src_embed = self.embedding(src) + self.pos_encoder(pos)
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        out = self.transformer(src_embed, src_mask)
        return self.fc_out(out)

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        
        for text in texts:
            encodings = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings['input_ids']))
            self.attn_masks.append(torch.tensor(encodings['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

def train(model, train_loader, optimizer, scheduler, criterion, device, scaler):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids, attention_mask = batch
        input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_k=50, device='cuda'):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    with torch.no_grad():
        for _ in range(max_length):
            outputs = model(input_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            probs = torch.softmax(top_k_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            next_token = top_k_indices.gather(-1, next_token)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(input_ids[0], skip_special_tokens=True)
    return generated_text

def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    BATCH_SIZE = 16
    EPOCHS = 60
    LEARNING_RATE = 3e-4
    WARMUP_STEPS = 500

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    vocab_size = tokenizer.vocab_size

    texts = get_ai_texts()

    full_dataset = TextDataset(texts, tokenizer)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, pin_memory=True, num_workers=2)

    model = TransformerModel(vocab_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

    scaler = torch.cuda.amp.GradScaler()

    os.makedirs("checkpoints", exist_ok=True)

    best_val_loss = float('inf')
    for epoch in range(EPOCHS):
        train_loss = train(model, train_loader, optimizer, scheduler, criterion, device, scaler)
        val_loss = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        checkpoint_path = f"checkpoints/model_checkpoint_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_checkpoint_path = checkpoint_path
            print(f"New best model saved: {best_checkpoint_path}")
        
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