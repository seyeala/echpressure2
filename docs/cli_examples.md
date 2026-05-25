# CLI Examples

## Macro windows

```bash
echopress detect-macro-windows \
  --dataset-root ./dataset \
  --align-table ./dataset/align.json \
  --output-dir ./runs/macro
```

## Echo peaks

```bash
echopress detect-echo-peaks \
  --detection-dir ./runs/macro \
  --output-dir ./runs/echo \
  --hilbert-frac 0.2 \
  --min-prominence-rel 0.08
```

## Postprocess

```bash
echopress postprocess-peak-windows \
  --echo-dir ./runs/echo \
  --output-dir ./runs/post \
  --max-echo-peak-order 3
```

## FFT

```bash
echopress fft-postprocessed \
  --postprocess-dir ./runs/post \
  --output-dir ./runs/fft \
  --fft-bins 256
```

## End-to-end (sequential)

Run all four commands in order. Each stage expects outputs from the previous stage.
