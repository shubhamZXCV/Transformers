from infer import greedy_decode , topk_sampling_decode , beam_search_decode
import torch
import torch.nn.functional as F
import sacrebleu

from config import get_config  , latest_weights_file_path
from train import get_model , get_ds

from pathlib import Path
import warnings

from tqdm import tqdm

def calculate_bleu(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device,
                  decoding_strategy="greedy", beam_size=5, top_k=10, num_examples=2):
    """
    Run inference on validation dataset and compute corpus BLEU score.
    decoding_strategy: "greedy", "beam", or "topk"
    """
    model.eval()
    all_references = []
    all_predictions = []
    count = 0

    with torch.no_grad():
        # Wrap the dataloader with tqdm
        for batch in tqdm(validation_ds, desc=f"Evaluating ({decoding_strategy})", unit="batch"):
            encoder_input = batch["encoder_input"].to(device)
            encoder_mask = batch["encoder_mask"].to(device)

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

            # --- Decode text ---
            target_text = batch["tgt_text"][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())

            # Collect for BLEU
            all_references.append([target_text])
            all_predictions.append(model_out_text)

            # Print first few examples only
            if count < num_examples:
                tqdm.write(f"SOURCE    : {batch['src_text'][0]}")
                tqdm.write(f"TARGET    : {target_text}")
                tqdm.write(f"PREDICTED : {model_out_text}")
                tqdm.write("-"*50)

            count += 1

    # --- Compute BLEU (corpus-level) ---
    bleu = sacrebleu.corpus_bleu(all_predictions, list(zip(*all_references)))
    print(f"\n✅ Corpus BLEU score ({decoding_strategy} decoding): {bleu.score:.2f}")

    return bleu.score


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

    # Run all decoding strategies
    print("\n--- Greedy Decoding ---")
    calculate_bleu(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                  config['seq_len'], device, decoding_strategy="greedy")

    print("\n--- Beam Search Decoding ---")
    calculate_bleu(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                  config['seq_len'], device, decoding_strategy="beam", beam_size=5)

    print("\n--- Top-k Sampling Decoding ---")
    calculate_bleu(model, val_dataloader, tokenizer_src, tokenizer_tgt,
                  config['seq_len'], device, decoding_strategy="topk", top_k=20)

