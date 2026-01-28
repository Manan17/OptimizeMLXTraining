"""
Baseline training with SmolLM (same as FLCE version for comparison).
"""
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load
from mlx_lm.tuner.trainer import TrainingArgs, train
from mlx_lm.tuner.utils import linear_to_lora_layers
from mlx_lm.tuner.datasets import TextDataset, CacheDataset
from datasets import load_dataset as hf_load_dataset

MODEL_PATH = "mlx-community/SmolLM-135M-fp16"
DATASET = "roneneldan/TinyStories"
OUTPUT_DIR = "./lora_tinystories"

SET_LORA = True
LORA_RANK = 8
LORA_LAYERS = 8

def main():
    model, tokenizer = load(MODEL_PATH)

    model.freeze()
    if SET_LORA:
        linear_to_lora_layers(model, LORA_LAYERS, {"rank": LORA_RANK, "scale": 16.0, "dropout": 0.0})

    raw_train = hf_load_dataset(DATASET, split="train[:1000]")
    raw_val = hf_load_dataset(DATASET, split="validation[:100]")

    train_data = [{"text": ex["text"]} for ex in raw_train]
    val_data = [{"text": ex["text"]} for ex in raw_val]

    dataset = CacheDataset(TextDataset(train_data, tokenizer))
    val_dataset = CacheDataset(TextDataset(val_data, tokenizer))

    optimizer = optim.Adam(learning_rate=1e-4)

    training_args = TrainingArgs(
        batch_size=4,
        iters=7,
        val_batches=5,
        steps_per_report=5,
        steps_per_eval=10,
        steps_per_save=20,
        adapter_file=OUTPUT_DIR + "/adapters.safetensors",
        max_seq_length=128,
    )

    train(
        model=model,
        optimizer=optimizer,
        args=training_args,
        train_dataset=dataset,
        val_dataset=val_dataset,
    )
    print(f"Weights saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
