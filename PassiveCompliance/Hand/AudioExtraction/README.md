# AudioExtraction

Extracts audio from the three guitar experiment recordings and normalises them
to a shared amplitude scale so RMS intensity is directly comparable across
stiffness conditions.

## Usage

Place the three video files in this folder, then run:

```bash
python extract_audio.py GUITAR_K1.MP4 GUITAR_K2.MP4 GUITAR_K3.MP4
```

All three files must be passed together so the global peak normalisation is
computed across all recordings.

**Output:** `GUITAR_K1.wav`, `GUITAR_K2.wav`, `GUITAR_K3.wav` — ready for
`plot_guitar.ipynb`.
