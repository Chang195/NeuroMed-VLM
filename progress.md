
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

## Day 3 — 2026-10-19

**Focus:** Streaming dataset sampling, biomedical tokenization, and PyTorch Dataset.

### Completed
- Sampled 1,000 ROCOv2 training examples using streaming buffer shuffle (`seed=42`, `buffer_size=5000`).
- Analyzed caption word counts and BiomedBERT token lengths.
- Calculated token length percentiles: P50 = 25.5, P90 = 54, P95 = 68, P99 = 94.08.
- Found that 57/1,000 captions exceed 64 tokens and 2/1,000 exceed 128 tokens.
- Tentatively selected `max_length=128` for text preprocessing.
- Loaded the BiomedBERT tokenizer and examined token IDs, special tokens, and attention masks.
- Successfully performed dynamic padding with a four-caption batch (`[4, 43]`).
- Implemented a custom PyTorch `MedicalDataset` with `__init__`, `__len__`, and `__getitem__`.
- Verified that the dataset contains 1,000 samples and correctly returns image, caption, and image ID.

### Next
- Implement a custom DataLoader and `collate_fn`.
- Convert PIL images to normalized RGB tensors.
- Tokenize and dynamically pad captions within batches.
- Validate image and text batch shapes before model training.