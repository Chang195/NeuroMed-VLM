
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

## Day 4 — 2026-10-20

**Focus:** Multimodal data preprocessing and PyTorch DataLoader.

### Completed
- Created `02_data_pipeline.ipynb` for reusable multimodal data processing.
- Implemented image preprocessing: RGB conversion, resizing to 224×224, tensor conversion, and ImageNet normalization.
- Integrated image transforms into a custom `MedicalDataset`.
- Implemented a custom `collate_fn` for image stacking and BiomedBERT batch tokenization with dynamic padding.
- Used `functools.partial` to bind the tokenizer to the collate function.
- Created a shuffled PyTorch DataLoader with `batch_size=4`.
- Validated image batches with shape `[4, 3, 224, 224]` and dynamically padded text batches.
- Successfully iterated through all 250 batches and 1,000 samples without shape assertion errors.

### Key Concepts Learned
- Sample-level preprocessing vs. batch-level processing
- `torch.stack()` vs. `torch.cat()`
- Token IDs, attention masks, and embedding inputs
- Dataset, DataLoader, and custom collate functions
- Dynamic padding and shuffled mini-batches

### Next Steps
- Explore DINOv2 and BiomedBERT model architectures.
- Load pretrained image and text encoders.
- Run initial forward passes and inspect embedding dimensions.
- Verify preprocessing compatibility with pretrained model configurations.

### Known Limitations
- Current image resizing may distort the original aspect ratio.
- Dataset sampling and preprocessing are not yet packaged as reusable Python modules.
- End-to-end model forward passes and image–text pairing integrity remain to be tested.

## Day 5 — 2026-10-21

**Focus:** DINOv2 and BiomedBERT architecture exploration, forward passes, and cross-modal projection.

### Completed
- Loaded pretrained DINOv2-Small (22.06M parameters).
- Examined DINOv2 configuration: 12 Transformer layers, 6 attention heads, hidden size 384, patch size 14.
- Successfully ran DINOv2 forward pass on a 224×224 medical image.
- Obtained image hidden states with shape `[1, 257, 384]`.
- Extracted CLS embedding `[1, 384]` and patch embeddings `[1, 256, 384]`.
- Loaded pretrained BiomedBERT-Base (109.48M parameters).
- Successfully ran BiomedBERT forward pass, obtaining hidden states `[1, 10, 768]`.
- Implemented CLS pooling and attention-mask-aware mean pooling.
- Implemented image projection `384 → 256` and text projection `768 → 256`.
- Applied L2 normalization and computed cosine similarity between image and text embeddings.

### Key Concepts Learned
- Self-supervised visual representation learning and teacher–student training
- Vision Transformer patch embeddings and CLS tokens
- Transformer layers vs. attention heads
- Model parameter scaling with hidden dimension
- `model.eval()`, `torch.no_grad()`, and `requires_grad=False`
- CLS pooling vs. masked mean pooling
- Projection heads, L2 normalization, and cosine similarity

### Experimental Results
- Image embedding: `[1, 256]`
- Text embedding: `[1, 256]`
- Both normalized embedding L2 norms: approximately 1
- Initial cosine similarity: approximately -0.0568 (untrained projection heads; not a meaningful alignment metric)

### Next Steps
- Organize the two encoders and projection heads into a reusable dual-encoder model.
- Construct a batch-level image–text similarity matrix.
- Learn and implement CLIP-style contrastive loss (InfoNCE).
- Establish an initial training baseline.

### Known Limitations
- Projection heads are randomly initialized and not yet trained.
- Encoder preprocessing and pooling choices require controlled evaluation.
- Current tests validate forward computation, not learned cross-modal alignment.