# LM_Dec_Long_TransformerXL

TransformerXL - Long family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Benchmarked**
- Failure class: none

## Provenance

- Source repository: (new) lucidrains/x-transformers 2.31.7
- Upstream / license: lucidrains/x-transformers (MIT), pip 2.31.7

## Differences from the source ([Unified] protocol)

- segments of 128 with 256 cached memories, relative position bias

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 19.16 M
- Causality check: pass (largest change of earlier outputs 0.0e+00)
- Overfit check (4 x 256 bytes): 5.720 -> 0.001 nats, pass
- LM-02 enwik8 (200M tokens): valid 1.462 / test 1.446 bits per byte
- LM-02 enwik8 (1B tokens): valid 1.371 / test 1.364 bits per byte
- Peak GPU memory at length 2048 (batch 1): train 735 MB, inference 78 MB
- Peak GPU memory at length 8192 (batch 1): train 2850 MB, inference 81 MB
- LM-05 recall `lm05v3`: 4 pairs @ 512: 0.20 (keys removed 0.20); 4 pairs @ 2048: 0.02 (keys removed 0.02); 16 pairs @ 512: 0.06 (keys removed 0.05); 16 pairs @ 2048: 0.02 (keys removed 0.02)
- LM-05 recall `lm05v3long`: 4 pairs @ 512: 0.25 (keys removed 0.25); 4 pairs @ 2048: 0.01 (keys removed 0.01); 4 pairs @ 8192: 0.02 (keys removed 0.02); 16 pairs @ 512: 0.07 (keys removed 0.06); 16 pairs @ 2048: 0.02 (keys removed 0.01); 16 pairs @ 8192: 0.01 (keys removed 0.02)
- LM-06 memory update `lm06`: @512: latest 0.12, unchanged 0.07, stale 0.08; @2048: latest 0.02, unchanged 0.02, stale 0.02

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy x-transformers==2.31.7
python run.py check xt_xl
python run.py lm02 xt_xl      # enwik8 is downloaded to data/ on first use
```
