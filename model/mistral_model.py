import requests
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
import math

@dataclass
class ModelConfig:
    d_model: int = 256
    vocab_size: int = 65
    n_layers: int = 6
    num_attention_heads: int = 8
    n_kv_heads: int = 2
    max_position_embeddings: int = 512
    attention_window: int = 128
    layer_norm_eps: float = 1e-12
    intermediate_size: int = 1024  # Added this parameter


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048):
        super(RotaryEmbedding, self).__init__()

        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))

        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len_cahed = max_position_embeddings

        t = torch.arange(self.max_seq_len_cahed).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    def forward(self, x, seq_len=None):
        if seq_len > self.max_seq_len_cahed:
            self._set_cos_sin_cache(seq_len)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(x.device),
            self.sin_cached[:, :, :seq_len, ...].to(x.device),
        )

    def _set_cos_sin_cache(self, seq_len):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached).type_as(self.inv_freq)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1).to(self.inv_freq.device)

        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])


def rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_attention_heads, n_kv_heads, max_position_embeddings=2048, attention_window=512):
        super(GroupedQueryAttention, self).__init__()

        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        self.n_kv_heads = n_kv_heads
        self.num_key_value_groups = num_attention_heads // n_kv_heads
        self.head_dim = d_model // num_attention_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)

        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings=max_position_embeddings)

        self.attention_window = attention_window

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        B, T, _ = hidden_states.shape

        query = self.q_proj(hidden_states).view(B, T, self.num_attention_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        cos, sin = self.rotary_emb(query, seq_len=T)

        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        if past_key_value is not None:
            key = torch.cat([past_key_value[0], key], dim=2)
            value = torch.cat([past_key_value[1], value], dim=2)
        
        past_key_value = (key, value)

        # Repeat k and v for each query in the group
        key = key.repeat_interleave(self.num_key_value_groups, dim=1)
        value = value.repeat_interleave(self.num_key_value_groups, dim=1)

        # apply sliding window attention
        attn_weights = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(self.head_dim)

        if self.attention_window < T:
            attn_weights = self._apply_sliding_window_attention(attn_weights)

        # apply attention mask
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # normalize attention weights
        attn_weights = F.softmax(attn_weights, dim=-1)

        output = torch.matmul(attn_weights, value)
        output = output.transpose(1, 2).contiguous().view(B, T, self.d_model)
        output = self.out_proj(output)

        return output, past_key_value

    def _apply_sliding_window_attention(self, attn_weights):
        # apply sliding window attention
        batch_size, num_heads, seq_len, _ = attn_weights.shape
        masked_attn_weights = attn_weights.clone()

        for i in range(seq_len):
            window_start = max(0, i - self.attention_window // 2)
            window_end = min(seq_len, i + self.attention_window // 2)

            masked_attn_weights[:, :, i, :window_start] = float('-inf')
            masked_attn_weights[:, :, i, window_end:] = float('-inf')

        return masked_attn_weights

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super(Block, self).__init__()

        self.attention = GroupedQueryAttention(config.d_model, config.num_attention_heads, config.n_kv_heads, config.max_position_embeddings, config.attention_window)

        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.d_model),
        )
        self.attention_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(self, hidden_states, attention_mask=None, past_key_value=None):
        attention_output, past_key_value = self.attention(
            self.attention_norm(hidden_states),
            attention_mask=attention_mask,
            past_key_value=past_key_value
        )
        hidden_states = hidden_states + attention_output # residual connection

        feed_forward_output = self.feed_forward(self.ffn_norm(hidden_states))

        hidden_states = hidden_states + feed_forward_output # residual connection

        return hidden_states, past_key_value
    

class MistralModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MistralModel, self).__init__()

        self.config = config
        self.embeddings = nn.Embedding(config.vocab_size, config.d_model)

        self.layers = nn.ModuleList([
            Block(config) for _ in range(config.n_layers)
        ])

        self.final_layer_norm = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        hidden_states = self.embeddings(input_ids)

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, past_key_value = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value
            )
            past_key_values[i] = past_key_value

        hidden_states = self.final_layer_norm(hidden_states)

        return hidden_states, past_key_values
    

class MistralForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super(MistralForCausalLM, self).__init__()

        self.model = MistralModel(config)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, input_ids, attention_mask=None, past_key_values=None):
        hidden_states, past_key_values = self.model(input_ids, attention_mask, past_key_values)
        logits = self.lm_head(hidden_states)

        return logits, past_key_values
    
url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
response = requests.get(url)
data = response.text

# Tokenize the dataset
chars = sorted(list(set(data)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
encoded_data = torch.tensor(encode(data), dtype=torch.long)

seq_length = 128
input_ids = torch.stack([encoded_data[i:i+seq_length] for i in range(0, len(encoded_data) - seq_length, seq_length)])
labels = torch.stack([encoded_data[i+1:i+1+seq_length] for i in range(0, len(encoded_data) - seq_length, seq_length)])

dataset = TensorDataset(input_ids, labels)
dataloader = DataLoader(dataset, batch_size=256, shuffle=True)

config = ModelConfig(
    d_model=256,
    vocab_size=vocab_size,
    n_layers=6,
    num_attention_heads=8,
    n_kv_heads=2,
    max_position_embeddings=512,
    attention_window=128,
    layer_norm_eps=1e-12
)

model = MistralForCausalLM(config)
model.train()
optim = torch.optim.AdamW(model.parameters(), lr=5e-5)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=50)

def generate_text(model, input_text, stoi, itos, max_length=100):
    # Ensure model is in evaluation mode
    model.eval()
    
    # Encode input text to tokens
    input_ids = torch.tensor([stoi[c] for c in input_text], dtype=torch.long).unsqueeze(0)  # Add batch dimension

    # Prepare past key values to speed up generation
    past_key_values = None

    generated_text = input_text

    with torch.no_grad():  # No need to compute gradients for generation
        for _ in range(max_length):
            # Forward pass through the model
            logits, past_key_values = model(input_ids, past_key_values=past_key_values)
            
            # Take the last token's logits and compute the next token
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).item()

            # Append next token to input_ids for the next step
            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=1)

            # Convert token id back to character
            next_char = itos[next_token_id]
            generated_text += next_char

            # Break if end of sequence token (if defined) is generated
            # if next_token_id == your_end_token_id:
            #     break

    return generated_text

for epoch in range(2):
    total_loss = 0
    total_correct = 0
    total_tokens = 0

    for batch in dataloader:
        input_ids, labels = batch

        logits, _ = model(input_ids)

        loss = F.cross_entropy(logits.view(-1, config.vocab_size), labels.view(-1))

        optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optim.step()

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=-1) == labels).sum().item()
        total_tokens += labels.numel()

        print(f"Epoch: {epoch}, Loss: {loss.item()}, Accuracy: {torch.sum(torch.argmax(logits, dim=-1) == labels) / len(labels)}")

    scheduler.step()
    print(f"Epoch: {epoch}, Loss: {total_loss / len(dataloader)}, Accuracy: {total_correct / total_tokens * 100:.2f}%")

    if (epoch + 1) % 10 == 0:
        sample_text = generate_text(model, "To be or not to be", stoi, itos, max_length=100)
        print(f"Sample generated text:\n{sample_text}\n")



# Example Usage
input_text = "To be"
generated_text = generate_text(model, input_text, stoi, itos)
print("Generated Text: ", generated_text)

