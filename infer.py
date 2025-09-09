import torch
import torch.nn.functional as F
from dataset import causal_mask
from config import get_config  , latest_weights_file_path
from train import get_model , get_ds
from pathlib import Path

import os
import warnings

def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # calculate output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0)



def beam_search_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt,
                       max_len, device, beam_size=5):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # Encode once
    encoder_output = model.encode(source, source_mask)

    # Each beam: (sequence tensor, score)
    beams = [(torch.tensor([[sos_idx]], dtype=source.dtype, device=device), 0.0)]

    for _ in range(max_len):
        new_beams = []
        for seq, score in beams:
            if seq[0, -1].item() == eos_idx:  # already ended
                new_beams.append((seq, score))
                continue

            # Build mask
            tgt_mask = causal_mask(seq.size(1)).type_as(source_mask).to(device)

            # Decode
            out = model.decode(encoder_output, source_mask, seq, tgt_mask)
            logits = model.project(out[:, -1])  # (1, vocab_size)
            log_probs = F.log_softmax(logits, dim=-1)

            # Get top beam_size expansions
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_size, dim=-1)

            for k in range(beam_size):
                next_token = topk_indices[0, k].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1)
                new_score = score + topk_log_probs[0, k].item()
                new_beams.append((new_seq, new_score))

        # Keep best beam_size
        beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_size]

        # If all beams ended with EOS → stop
        if all(seq[0, -1].item() == eos_idx for seq, _ in beams):
            break

    # Pick best beam
    best_seq, _ = max(beams, key=lambda x: x[1])
    return best_seq.squeeze(0)

def topk_sampling_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt,
                         max_len, device, k=10):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    encoder_output = model.encode(source, source_mask)

    # Start sequence with SOS
    seq = torch.tensor([[sos_idx]], dtype=source.dtype, device=device)

    for _ in range(max_len):
        tgt_mask = causal_mask(seq.size(1)).type_as(source_mask).to(device)

        out = model.decode(encoder_output, source_mask, seq, tgt_mask)
        logits = model.project(out[:, -1])  # (1, vocab_size)
        probs = F.softmax(logits, dim=-1)

        # Restrict to top-k
        topk_probs, topk_indices = torch.topk(probs, k, dim=-1)

        # Normalize to make a proper distribution
        topk_probs = topk_probs / torch.sum(topk_probs, dim=-1, keepdim=True)

        # Sample one token
        next_token = torch.multinomial(topk_probs, 1)
        next_word = topk_indices[0, next_token.item()].unsqueeze(0).unsqueeze(0)

        seq = torch.cat([seq, next_word], dim=1)

        if next_word.item() == eos_idx:
            break

    return seq.squeeze(0)

def run_inference(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device,
                  print_msg=print, num_examples=2, decoding_strategy="greedy",
                  beam_size=5, top_k=10):
    """
    Run inference on validation dataset with different decoding strategies.
    
    decoding_strategy: "greedy", "beam", or "topk"
    """
    model.eval()
    count = 0

    try:
        with os.popen('stty size', 'r') as console:
            _, console_width = console.read().split()
            console_width = int(console_width)
    except:
        console_width = 80  # fallback

    with torch.no_grad():
        for batch in validation_ds:
            count += 1
            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(device)    # (b, 1, 1, seq_len)

            assert encoder_input.size(0) == 1, "Batch size must be 1 for inference"

            # --- Choose decoding strategy ---
            if decoding_strategy == "greedy":
                model_out = greedy_decode(model, encoder_input, encoder_mask,
                                          tokenizer_src, tokenizer_tgt, max_len, device)
            elif decoding_strategy == "beam":
                model_out = beam_search_decode(model, encoder_input, encoder_mask,
                                               tokenizer_src, tokenizer_tgt, max_len, device,
                                               beam_size=beam_size)
            elif decoding_strategy == "topk":
                model_out = topk_sampling_decode(model, encoder_input, encoder_mask,
                                                 tokenizer_src, tokenizer_tgt, max_len, device,
                                                 k=top_k)
            else:
                raise ValueError(f"Unknown decoding strategy: {decoding_strategy}")

            # --- Convert to text ---
            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # --- Print nicely ---
            print_msg('-'*console_width)
            print_msg(f"{f'SOURCE: ':>12}{source_text}")
            print_msg(f"{f'TARGET: ':>12}{target_text}")
            print_msg(f"{f'PREDICTED: ':>12}{model_out_text}")

            if count == num_examples:
                print_msg('-'*console_width)
                break


def load_model_and_data(config):
    """Load model and validation dataset only once."""
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    device = torch.device(device)
    print("Using device:", device)

    # Load dataset (only validation for inference)
    _, val_dataloader, _, tokenizer_src, tokenizer_tgt = get_ds(config)

    # Load model
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    # Load latest checkpoint
    model_filename = latest_weights_file_path(config)
    if model_filename and Path(model_filename).exists():
        print(f"Loading model checkpoint: {model_filename}")
        state = torch.load(model_filename, map_location=device)
        model.load_state_dict(state['model_state_dict'])
    else:
        raise FileNotFoundError("❌ No saved model checkpoint found!")

    return model, val_dataloader, tokenizer_src, tokenizer_tgt, device


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()

    # Load once
    model, val_dataloader, tokenizer_src, tokenizer_tgt, device = load_model_and_data(config)

    # Run different decoding strategies without reloading
    print("\n--- Greedy Decoding ---")
    run_inference(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                  config['seq_len'], device, decoding_strategy="greedy", num_examples=3)

    print("\n--- Beam Search Decoding ---")
    run_inference(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                  config['seq_len'], device, decoding_strategy="beam", beam_size=5, num_examples=3)

    print("\n--- Top-k Sampling Decoding ---")
    run_inference(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                  config['seq_len'], device, decoding_strategy="topk", top_k=20, num_examples=3)



