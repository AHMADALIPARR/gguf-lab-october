# sandbox/ — isolated benchmark environment

Benchmarks run here, not in the repo root. Isolation rules:

- **Inputs** are versioned (`inputs/prompts.json`). Change prompts → new version.
- **Outputs** go to `outputs/` (gitignored). Reports never pollute the repo.
- **Config** (`config.json`) pins resource limits, runtime flags, model paths.
- **Fail closed**: missing model, bad SHA256, or low RAM aborts before inference.
- **MoE router** (`moe/`) does sparse expert selection via embeddings.

## Layout

| Path | Role |
|---|---|
| `config.json` | Resource limits, runtime flags, model paths |
| `inputs/prompts.json` | Versioned benchmark prompts |
| `outputs/` | Reports (gitignored) |
| `run.py` | Sandboxed benchmark runner |
| `moe/experts.json` | Expert definitions + anchor prompts |
| `moe/router.py` | Embedding-driven sparse router |
| `moe/centroids.json` | Built anchor centroids (generated) |

## Run benchmarks

```sh
python3 sandbox/run.py \
  --runtime /path/to/llama-cli \
  --model models/mellum-E8-experimental.gguf \
  --expected-sha256 <sha256> \
  --tag mellum-e8
```

## Run MoE router

```sh
# Terminal 1: embedding server
llama-server -m models/embeddinggemma-2-UD-Q4_K_XL.gguf --embedding -c 512

# Terminal 2: build centroids once
python3 sandbox/moe/router.py --build-centroids

# Route a query
python3 sandbox/moe/router.py --query "Write a quicksort in Python" --top-k 1

# Route and run the selected expert
python3 sandbox/moe/router.py --query "Write a quicksort in Python" \
  --run --runtime /path/to/llama-cli
```

## Design notes

The router uses **sparse activation**: only the top-k experts (by cosine
similarity between query embedding and anchor centroids) are loaded.
EmbeddingGemma (768-dim) is the router model — small, fast, CPU-friendly.

Expert specialties are defined by anchor prompts in `experts.json`.
To add an expert: append its definition, rebuild centroids.
