import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import GPT2Tokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import os
import warnings
from datasets import load_dataset
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time
import traceback
import random

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
        
        for text in tqdm(texts, desc="Tokenizing texts"):
            encodings = tokenizer(text, truncation=True, max_length=max_length, padding="max_length")
            self.input_ids.append(torch.tensor(encodings['input_ids']))
            self.attn_masks.append(torch.tensor(encodings['attention_mask']))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

class StreamingTextDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, split='train', max_length=512, split_ratio=0.9):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.split = split
        self.split_ratio = split_ratio

    def __iter__(self):
        for item in self.dataset[self.split]:
            if self.split == 'train' and random.random() > self.split_ratio:
                continue
            if self.split == 'val' and random.random() <= self.split_ratio:
                continue
            text = f"Instruction: {item['instruction']}\nInput: {item['input']}\nOutput: {item['output']}"
            encodings = self.tokenizer(text, truncation=True, max_length=self.max_length, padding="max_length")
            yield {
                'input_ids': torch.tensor(encodings['input_ids']),
                'attention_mask': torch.tensor(encodings['attention_mask'])
            }


def train(model, train_loader, optimizer, scheduler, criterion, device, scaler, gui):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
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
        gui.update_loss(loss.item())
    return total_loss / len(train_loader)

def evaluate(model, val_loader, criterion, device, gui):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids)
            loss = criterion(outputs.view(-1, outputs.size(-1)), input_ids.view(-1))
            total_loss += loss.item()
            gui.update_loss(loss.item())
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


class TrainingGUI:
    def __init__(self, master):
        self.master = master
        master.title("AI Training Progress")
        master.geometry("600x400")

        self.progress_var = tk.DoubleVar()
        self.epoch_var = tk.StringVar()
        self.time_var = tk.StringVar()
        self.loss_var = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        # Progress bar
        ttk.Label(self.master, text="Training Progress:").pack(pady=10)
        self.progress_bar = ttk.Progressbar(self.master, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=20, pady=5)

        # Info labels
        info_frame = ttk.Frame(self.master)
        info_frame.pack(fill=tk.X, padx=20, pady=10)

        ttk.Label(info_frame, text="Epoch:").grid(row=0, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.epoch_var).grid(row=0, column=1, sticky="w")

        ttk.Label(info_frame, text="Estimated Time:").grid(row=1, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.time_var).grid(row=1, column=1, sticky="w")

        ttk.Label(info_frame, text="Current Loss:").grid(row=2, column=0, sticky="w")
        ttk.Label(info_frame, textvariable=self.loss_var).grid(row=2, column=1, sticky="w")

        # Console output
        ttk.Label(self.master, text="Console Output:").pack(pady=10)
        self.console = scrolledtext.ScrolledText(self.master, height=10)
        self.console.pack(fill=tk.BOTH, expand=True, padx=20, pady=5)
        self.console.tag_config('error', foreground='red')

    def update_progress(self, progress):
        self.progress_var.set(progress)

    def update_epoch(self, epoch, total_epochs):
        self.epoch_var.set(f"{epoch}/{total_epochs}")

    def update_time(self, time_left):
        self.time_var.set(time_left)

    def update_loss(self, loss):
        self.loss_var.set(f"{loss:.4f}")

    def write_to_console(self, message, error=False):
        tag = 'error' if error else 'normal'
        self.console.insert(tk.END, message + "\n", tag)
        self.console.see(tk.END)

def main():
    root = tk.Tk()
    gui = TrainingGUI(root)

    def training_thread():
        try:
            torch.manual_seed(42)
            torch.cuda.manual_seed_all(42)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            gui.write_to_console(f"Using device: {device}")

            if device.type == 'cuda':
                gui.write_to_console(f"GPU: {torch.cuda.get_device_name(0)}")
                gui.write_to_console(f"CUDA Version: {torch.version.cuda}")

            BATCH_SIZE = 16
            EPOCHS = 100
            LEARNING_RATE = 3e-4
            WARMUP_STEPS = 500

            tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            tokenizer.pad_token = tokenizer.eos_token
            vocab_size = tokenizer.vocab_size

            gui.write_to_console("Loading dataset...")
            ds = load_dataset("Replete-AI/Everything_Instruct_8k_context_filtered", cache_dir="./my_dataset_cache", streaming=True)
            
            gui.write_to_console("Preparing dataset...")
            train_dataset = StreamingTextDataset(ds, tokenizer, split='train')
            val_dataset = StreamingTextDataset(ds, tokenizer, split='val')

            train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=2)
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=2)

            model = TransformerModel(vocab_size).to(device)

            criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
            optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
            
            total_steps = 1000000 // BATCH_SIZE * EPOCHS  # Approximation
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS, num_training_steps=total_steps)

            scaler = torch.cuda.amp.GradScaler()

            os.makedirs("checkpoints", exist_ok=True)

            best_val_loss = float('inf')
            start_time = time.time()
            for epoch in range(EPOCHS):
                train_loss = train(model, train_loader, optimizer, scheduler, criterion, device, scaler, gui)
                val_loss = evaluate(model, val_loader, criterion, device, gui)
                
                gui.update_epoch(epoch + 1, EPOCHS)
                gui.update_progress((epoch + 1) / EPOCHS * 100)
                gui.update_loss(val_loss)
                
                elapsed_time = time.time() - start_time
                estimated_total_time = elapsed_time / (epoch + 1) * EPOCHS
                time_left = estimated_total_time - elapsed_time
                gui.update_time(f"{time_left/3600:.2f} hours")

                gui.write_to_console(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

                checkpoint_path = f"checkpoints/model_checkpoint_epoch_{epoch+1}.pth"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, checkpoint_path)
                gui.write_to_console(f"Checkpoint saved: {checkpoint_path}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_checkpoint_path = checkpoint_path
                    gui.write_to_console(f"New best model saved: {best_checkpoint_path}")
                
                if epoch > 10 and val_loss > best_val_loss:
                    gui.write_to_console("Early stopping")
                    break

            gui.write_to_console("Training completed!")
            checkpoint = torch.load(best_checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            prompt = "Instruction: Explain what artificial intelligence is\nInput: \nOutput:"
            generated_text = generate_text(model, tokenizer, prompt, device=device)
            gui.write_to_console(f"Generated text: {generated_text}")

        except Exception as e:
            error_message = f"An error occurred: {str(e)}\n{traceback.format_exc()}"
            gui.write_to_console(error_message, error=True)

    # Start the training in a separate thread
    threading.Thread(target=training_thread, daemon=True).start()

    root.mainloop()

if __name__ == "__main__":
    main()