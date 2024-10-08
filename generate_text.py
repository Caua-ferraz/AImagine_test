import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer
import tkinter as tk
from tkinter import ttk, scrolledtext
import warnings

warnings.filterwarnings("ignore")

# TransformerModel class definition (same as before)
class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=384, nhead=6, num_layers=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = nn.Embedding(1024, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=4*d_model, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, past_key_values=None):
        pos = torch.arange(0, src.size(1)).unsqueeze(0).repeat(src.size(0), 1).to(src.device)
        src_embed = self.embedding(src) + self.pos_encoder(pos)
        src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        out = self.transformer(src_embed, src_mask)
        logits = self.fc_out(out)
        return logits  # We're not returning past_key_values as they're not used in this simple model

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# generate_text function (same as before)
def generate_text(model, tokenizer, prompt, max_length=100, temperature=0.7, top_p=0.9, repetition_penalty=1.2, device='cuda'):
    model.eval()
    input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
    
    generated_ids = input_ids
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(generated_ids)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Apply repetition penalty
            for i in range(generated_ids.shape[1]):
                next_token_logits[0, generated_ids[0, i]] /= repetition_penalty
            
            # Apply nucleus (top-p) sampling
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[:, indices_to_remove] = -float('inf')
            
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated_ids = torch.cat([generated_ids, next_token], dim=-1)
            
            if next_token.item() == tokenizer.eos_token_id:
                break
    
    generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return generated_text

class TextGenerationGUI:
    def __init__(self, master):
        self.master = master
        master.title("AI Text Generator")
        master.geometry("600x400")

        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.create_widgets()
        self.load_model()

    def create_widgets(self):
        # Prompt input
        ttk.Label(self.master, text="Enter prompt:").pack(pady=5)
        self.prompt_entry = ttk.Entry(self.master, width=50)
        self.prompt_entry.pack(pady=5)
        self.prompt_entry.insert(0, "Artificial intelligence is")

        # Parameters
        param_frame = ttk.Frame(self.master)
        param_frame.pack(pady=10)

        ttk.Label(param_frame, text="Temperature:").grid(row=0, column=0, padx=5)
        self.temperature_entry = ttk.Entry(param_frame, width=10)
        self.temperature_entry.grid(row=0, column=1, padx=5)
        self.temperature_entry.insert(0, "1.0")

        ttk.Label(param_frame, text="Top-p:").grid(row=0, column=2, padx=5)
        self.top_p_entry = ttk.Entry(param_frame, width=10)
        self.top_p_entry.grid(row=0, column=3, padx=5)
        self.top_p_entry.insert(0, "0.9")

        ttk.Label(param_frame, text="Max Length:").grid(row=0, column=4, padx=5)
        self.max_length_entry = ttk.Entry(param_frame, width=10)
        self.max_length_entry.grid(row=0, column=5, padx=5)
        self.max_length_entry.insert(0, "100")

        # Generate button
        self.generate_button = ttk.Button(self.master, text="Generate Text", command=self.generate)
        self.generate_button.pack(pady=10)

        # Output text
        self.output_text = scrolledtext.ScrolledText(self.master, wrap=tk.WORD, width=70, height=10)
        self.output_text.pack(pady=10)

    def load_model(self):
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
            self.tokenizer.pad_token = self.tokenizer.eos_token
            vocab_size = self.tokenizer.vocab_size

            self.model = TransformerModel(vocab_size).to(self.device)
            checkpoint = torch.load("model_checkpoint_epoch_20.pth", map_location=self.device) # chose model epoch
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()

            self.output_text.insert(tk.END, "Model loaded successfully.\n")
        except Exception as e:
            self.output_text.insert(tk.END, f"Error loading model: {str(e)}\n")
            self.generate_button.config(state='disabled')

    def generate(self):
        prompt = self.prompt_entry.get()
        try:
            temperature = float(self.temperature_entry.get())
            top_p = float(self.top_p_entry.get())
            max_length = int(self.max_length_entry.get())
        except ValueError:
            self.output_text.insert(tk.END, "Invalid parameter values. Using defaults.\n")
            temperature, top_p, max_length = 0.7, 0.9, 100

        self.output_text.delete('1.0', tk.END)
        self.output_text.insert(tk.END, f"Prompt: {prompt}\n\n")
        self.output_text.insert(tk.END, "Generating text...\n\n")
        self.master.update()

        generated_text = generate_text(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.2,
            device=self.device
        )
        
        self.output_text.insert(tk.END, f"Generated text: {generated_text}\n")

def main():
    root = tk.Tk()
    app = TextGenerationGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()