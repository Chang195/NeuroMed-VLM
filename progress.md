
# NeuroMed-VLM Research Progress

## Day 1 — 2026-10-17

### Completed
- Conda environment: Python 3.11
- PyTorch CUDA verified on RTX 2060
- VS Code Jupyter Notebook configured
- ROCOv2 streaming dataset successfully loaded
- Medical image visualization
- Explored dataset schema, captions, CUI, PIL Image
- Inspected first 10 samples
- Average caption length: 16.1 words (manually verified)

### Next — Day 2
- Reopen explore_data.ipynb
- Reload ROCOv2 streaming dataset
- Analyze a larger sample of records
- Plot caption-length distribution
- Examine image sizes and CUI counts

### Environment
- Project: G:\NeuroMed-VLM
- Python: G:\NeuroMed-VLM\.conda\python.exe
- GPU: NVIDIA RTX 2060 (6 GB)


## Day 2 — 2026-10-18

### Completed
- Initialized Git repository and connected GitHub.
- Created .gitignore to exclude environments, data and model weights.
- Loaded first 100 ROCOv2 training samples.
- Constructed a Pandas DataFrame with 6 metadata fields.
- Analyzed caption word counts and quantiles.
- Plotted caption length histogram.
- Analyzed CUI counts and image color modes.
- Plotted image width vs. height.

### Preliminary Findings (n=100)
- Caption median: 16 words
- Caption range: 1–105 words
- Caption 95th percentile: 50.05 words
- CUI counts: 1 (80), 2 (3), 3 (17)
- Image modes: RGB (75), L (25)
- Width range: 307–1592 pixels
- Height range: 224–1205 pixels

### Limitations
- Only the first 100 sequential training samples were examined.
- Findings are not necessarily representative of the entire dataset.
- Caption lengths are word counts, not BiomedBERT token counts.

### Next
- Design a larger, more representative dataset audit.
- Examine caption tokenization and image preprocessing.
- Prepare reusable data processing scripts.
