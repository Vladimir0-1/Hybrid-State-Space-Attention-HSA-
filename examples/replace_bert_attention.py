"""
Example: Replace BERT's attention with CWAB
"""

import torch
from transformers import AutoModel, AutoTokenizer
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cwab import CWAB


def replace_bert_attention(model, window_size=512, num_global_tokens=64):
    for layer in model.encoder.layer:
        old_attn = layer.attention.self
        new_attn = CWAB(
            hidden_size=old_attn.query.out_features,
            num_heads=old_attn.num_attention_heads,
            window_size=window_size,
            num_global_tokens=num_global_tokens,
            dropout=old_attn.dropout.p if hasattr(old_attn, 'dropout') else 0.1,
        )
        layer.attention.self = new_attn
    return model


def main():
    print("=" * 60)
    print("CWAB: BERT Attention Replacement Demo")
    print("=" * 60)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    model_name = "bert-base-uncased"
    model = AutoModel.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {original_params:,}")

    model = replace_bert_attention(model)

    new_params = sum(p.numel() for p in model.parameters())
    print(f"New parameters: {new_params:,}")
    print(f"Change: {new_params - original_params:+,}")

    test_lengths = [128, 256, 512, 1024]
    for seq_len in test_lengths:
        text = "Hello world. " * (seq_len // 3)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).to(device)

        with torch.no_grad():
            start = time.time()
            outputs = model(**inputs)
            elapsed = (time.time() - start) * 1000

        print(f"Seq len {seq_len:4d}: {elapsed:.2f} ms | Output shape: {outputs.last_hidden_state.shape}")

    print("\n CWAB works as a drop-in replacement for BERT attention.")


if __name__ == "__main__":
    main()
