"""
Useful Links:

langchain_huggingface.HuggingFaceEmbeddings:

https://python.langchain.com/api_reference/huggingface/embeddings/langchain_huggingface.embeddings.huggingface.HuggingFaceEmbeddings.html

HuggingFaceEmbeddings calls on SentenceTransformer underneath the hood:

https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/SentenceTransformer.py

Note that the file only supports single-gpu. To run multiple files, invoke
this script with different gpus by running

```bash
export CUDA_VISIBLE_DEVICES=#
```

to enable parallelism.

Example usage:

python main.py \
    --model_name "Samsoup/Llama-3.2-3B-Instruct-FakeReviews" \
    --file_path "/work/06782/ysu707/ls6/ComputeEncodings/data/FakeReviews/train.csv" \
    --prompt "/work/06782/ysu707/ls6/ComputeEncodings/data/FakeReviews/prompts/zero_shot.txt" \
    --output_filename "train" \
    --output_dir "/work/06782/ysu707/ls6/ComputeEncodings/data/FakeReviews/data/FakeReviews/embeddings" \
    --batch_size 32 \
    --normalize_embeddings True \
    --cache_dir "/work/06782/ysu707/ls6/.cache"
"""

import os
import argparse
import pandas as pd
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
import torch


def load_embedding_model(
    model_name: str, cache_dir: str, batch_size: int, normalize_embeddings: bool
):
    # Set environment variables for cache
    os.environ["HF_HOME"] = cache_dir
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir

    # Define model and encoding arguments
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    encode_kwargs = {
        "normalize_embeddings": normalize_embeddings,
        "batch_size": batch_size,
    }

    # Initialize HuggingFaceEmbeddings
    embedding_model = HuggingFaceEmbeddings(
        model_name=model_name,
        cache_folder=cache_dir,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
        multi_process=False,
        show_progress=True,
    )

    # Set pad_token to eos_token if it's not already set
    if embedding_model.client.tokenizer.pad_token is None:
        embedding_model.client.tokenizer.pad_token = (
            embedding_model.client.tokenizer.eos_token
        )

    return embedding_model


# Function to prepend the prompt to the input text
def get_prompt_with_input(prompt: str, inputs: list):
    for inp in inputs:
        yield f"{prompt}\nInput:{inp}"


# Function to save embeddings
def save_embeddings(embeddings, output_path):
    np.save(output_path, embeddings)
    print(f"Embeddings saved to {output_path}")


def main(args):
    # Load the dataset
    df = pd.read_csv(args.file_path)
    texts = df["text"].tolist()

    # Load prompt
    if os.path.exists(args.prompt):
        with open(args.prompt, "r") as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt

    # Prepend prompt to texts
    texts_with_prompt = list(get_prompt_with_input(prompt, texts))

    # Load the embedding model
    embedding_model = load_embedding_model(
        model_name=args.model_name,
        cache_dir=args.cache_dir,
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize_embeddings,
    )

    # Compute embeddings
    embeddings = embedding_model.embed_documents(texts_with_prompt)

    # Convert embeddings to a numpy array and save
    embeddings_np = np.array(embeddings)
    output_path = os.path.join(args.output_dir, f"{args.output_filename}.npy")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    save_embeddings(embeddings_np, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings using HuggingFaceEmbeddings"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name to use for embeddings",
    )
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to the file containing the 'text' column",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Prompt string or path to prompt file",
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        required=True,
        help="Name of the output file. Will automatically append .npy to this",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save the embeddings",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for embedding computation",
    )
    parser.add_argument(
        "--normalize_embeddings",
        type=bool,
        default=True,
        help="Whether to normalize embeddings",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/work/06782/ysu707/ls6/.cache",
        help="Cache directory for model",
    )

    args = parser.parse_args()
    main(args)
