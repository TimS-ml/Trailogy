# Models/

MLX model files for the Kokoro TTS engine. **This directory ships empty in
git** — contents are downloaded by `scripts/fetch-models.sh`.

After fetching:

```
kokoro-v1_0.safetensors   ~600 MB   neural network weights (MLX format)
voices.npz                ~30 MB    28 pretrained voice embeddings
```

Source: [mlalma/KokoroTestApp Git LFS](https://github.com/mlalma/KokoroTestApp/tree/main/Resources).
The fetch script bypasses Git LFS by hitting GitHub's `raw/` URL, which
auto-redirects to the LFS storage host.

These files are loaded at app launch by `KokoroTTS(modelPath:)` (model) and
`NpyzReader.read(fileFromPath:)` (voices). MLX compiles graph kernels on
first use, so the first synthesis call takes longer than subsequent ones.

To re-download:

```
bash scripts/fetch-models.sh
```
