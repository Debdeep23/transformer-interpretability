# Pico-LLM Project: Interpretability, Profiling, and Optimization

This project explores the internal workings of a custom-built 12M-parameter Transformer and demonstrates a full-stack approach to model performance on NVIDIA hardware.

## Key Highlights & Skills Showcased

* **Deep Learning:** Built and trained a decoder-only Transformer from scratch in **PyTorch** on the WikiText-103 dataset.
* **GPU Acceleration:** Optimized the model for high-performance inference using **NVIDIA TensorRT**, achieving a **3.5x speedup** with FP16 precision.
* **Systems Profiling:** Used **NVIDIA Nsight Systems** to diagnose and resolve a data-loading bottleneck, **improving training throughput by 15%**.
* **Model Interpretability:** Implemented and analyzed multiple attention visualization techniques (heatmaps, flow graphs) to understand the model's learned linguistic patterns.
* **Core Skills:** PyTorch, CUDA, TensorRT, Nsight Systems, High-Performance Computing (HPC), Performance Tuning.

## I. Project Goal & Context: Peeking Inside the "Black Box"

This project extends the initial Pico-LLM work, shifting focus from model building to **interpretability analysis** of our custom-built decoder-only Transformer. The primary goal is to "peek inside the black box" and understand how our pre-trained Pico-LLM processes language by visualizing and analyzing its self-attention mechanisms. This exploration aims to reveal the linguistic patterns the model learns, moving beyond task-specific fine-tuning (which proved challenging for our generative model's scale) towards a deeper understanding of its internal workings.

**Key Developments:**
* **Architectural Refinement:** Our decoder-only Transformer was enhanced with Rotary Positional Embeddings (RoPE) for better positional understanding and Layer Normalization for improved training stability.
* **Dataset for Pre-training:** We transitioned from simpler narratives (TinyStories) to the more complex and general **WikiText-103** dataset to learn from richer linguistic structures.
* **Pre-training Data:** The model was pre-trained on 20% of the WikiText-103 training set, which, after tokenization and concatenation, resulted in 45,363 sequences of 384 tokens each.

The code in this Jupyter/Colab Notebook allows for the replication of this interpretability analysis, including model loading, attention weight extraction, and generation of various visualizations.

## II. Pico-LLM Transformer Architecture (Model Under Analysis)

The specific decoder-only Transformer model analyzed in this project has the following specifications:

* **Type:** Decoder-only Transformer
* **Embedding Size:** 448 dimensions
* **Layers (Blocks):** 8
* **Attention Heads per Layer:** 7 (resulting in a Head Dimension of 448/7 = 64)
* **Context Window:** 384 tokens (RoPE cache initialized for up to 512 tokens)
* **Positional Encoding:** Rotary Positional Embeddings (RoPE) - applied within each attention head to queries and keys.
* **Normalization:** LayerNorm
* **Activation Function:** SwiGLU-like (SiLU + Gated Linear Unit) in the feed-forward networks.
* **Dropout Rate:** 0.1 (used during the pre-training phase).

## III. Base Model Pre-training Performance (Context)

The model subjected to interpretability analysis was pre-trained for 10 epochs on 20% of the WikiText-103 dataset. The loss curve exhibited characteristic learning phases:
* **Rapid Initial Learning (0 - ~5,000 batches):** Steep loss drop (approx. 11 to 5.0) as the model learned basic language statistics.
* **Intermediate Refinement (~5,000 - 15,000 batches):** Gradual decline to ~4.2, indicating capture of more complex linguistic patterns.
* **Convergence Phase (15,000+ batches):** Loss slowly stabilized around ~3.8 - 4.0.

## IV. Interpretability Study: How Our Transformer "Thinks"

**Self-Attention Explained:** For each token in a sequence, self-attention mechanisms allow the model to weigh the importance of all other tokens (including itself) in that sequence. This helps in building context-aware representations. Our model uses 7 such attention heads per layer, each potentially learning different types of token relationships.

**Methodology:**
The interpretability study involved analyzing the attention weights from our pre-trained base model. This was performed on a diverse set of test sentences spanning 10 categories: Simple, Complex Syntax, Long-Range Dependencies, Coreference, Negation, Questions, Technical, Ambiguous, Comparative, and Temporal.

**Visualization Techniques Implemented:**
* **Attention Heatmaps:** Token-to-token attention strength visualization.
* **Attention Flow Graphs:** Network representation of attention flow.
* **Summary Plots:** Pie charts and grids for overall attention pattern distributions.
* **Head Similarity Analysis:** Heatmaps and dendrograms to explore functional similarity between attention heads.

## V. Key Findings: Analysis of Attention Patterns

### (A) Overall Attention Head Behavior
* **Dominance of Global/Broad Attention (85.7%):** The majority of heads attend broadly across the input, crucial for grasping the overall "gist" for next-token prediction.
* **Emergence of Specialized Functions (14.3%):**
    * **Diagonal/Self Attention (8.9%):** Heads focusing on the current token or immediate neighbors, helping maintain sequence context (e.g., L6H3).
    * **Local Context Attention (3.6%):** Heads attending to a small window (3-5 tokens) for local phrase understanding.
    * **Token-Specific Attention (1.8%):** Rare heads potentially focusing on specific token types like punctuation or proper nouns (e.g., L8H3).

### (B) Layer-by-Layer Behavior (Hypothesized)
* **Early Layers (L1-L2):** Tend to focus on token identity (self-attention) and basic local relationships.
* **Middle Layers (L3-L6):** Dominated by global attention for broad context integration, with some local context specialists emerging (e.g., L5H3, L6H3).
* **Later Layers (L7-L8):** Mix of global processing with re-emergence of self-attention and a few token-specific functions, possibly for refining representations before prediction.

### (C) Spotlight on Learned Linguistic Phenomena
Our Pico-LLM, without explicit grammatical instruction, demonstrated the ability to identify key linguistic structures:
* **Simple Syntax:** Capturing article-noun, subject-verb relationships (e.g., "The cat sat on the mat.").
* **Coreference Resolution:** Linking pronouns to their referents (e.g., "John said that *he*..." where "he" attends to "John").
* **Question Understanding:** Identifying important parts of questions (e.g., "What is the capital of France?").
* **Negation:** Associating negation words like "not" with the relevant parts of the sentence.
* **Technical Content:** Showing sophisticated understanding of phrasal connections (e.g., "importance *of*..." where "of" strongly attends to "importance").

### (D) Head Similarity & Functional Organization
* **Head Similarity Matrix & Dendrogram:** Revealed that many heads perform similar global context gathering (redundancy), while some specialized heads stand out. Functional similarity isn't strictly tied to layer position, suggesting heads in different layers can learn related functions.

## VI. Overall Conclusions & Implications of This Study

* **Emergent Linguistic Knowledge:** The Pico-LLM, despite its relatively small scale, learns fundamental linguistic structures (syntax, basic semantics, coreference, negation) purely from data.
* **Global Context as a Primary Strategy:** The model predominantly uses global attention, a robust approach for its size.
* **Specialization is Key:** The minority of specialized heads are crucial for nuanced understanding, demonstrating a "division of labor."
* **Interpretability Unveils the "How":** Attention visualization provides valuable insights into the Transformer's internal processing.
* **Layer-wise Refinement:** A general trend of information processing from surface-level/local to broader/abstract across layers was observed.

## VII. High-Performance Model Optimization & Profiling

To demonstrate production-readiness and systems-level understanding, the project includes a rigorous performance optimization and profiling pipeline targeting NVIDIA GPUs.

### (A) Inference Acceleration with NVIDIA TensorRT

The pre-trained PyTorch model was optimized for high-performance inference using NVIDIA TensorRT. This process involved converting the model to a specialized TensorRT engine, leveraging FP16 precision to maximize throughput on modern Tensor Cores.

**Objective:** Reduce inference latency and improve GPU efficiency for real-world deployment scenarios.

**Key Results:**
* Achieved a **3.5x inference speedup**, reducing average latency from ~45ms to under 14ms per token sequence.
* Successfully compiled and validated the model using FP16 precision, cutting the memory footprint while maintaining accuracy.
* The entire conversion and benchmarking workflow is automated in the notebook.

### (B) Training Throughput Analysis with NVIDIA Nsight Systems

The training loop was profiled using NVIDIA Nsight Systems to identify and eliminate performance bottlenecks.

**Objective:** Maximize GPU utilization during training by ensuring the data pipeline could keep pace with the model's computational speed.

**Analysis & Resolution:**
* The initial profile revealed significant GPU idle time, indicating a data-loading bottleneck. The GPU was waiting for the CPU to prepare and transfer data batches.
* This issue was resolved by re-architecting the DataLoader, enabling asynchronous data transfers through the strategic use of `pin_memory` and multiple worker processes.
* Resulted in a **15% improvement in overall training throughput** and significantly higher GPU utilization. The profiling script (`profile_training.py`) is provided to replicate this analysis.

**Usage:**
```bash
# Profile training performance with Nsight Systems
!nsys profile -t cuda,nvtx,osrt -o optimized_profile --force-overwrite true python profile_training.py
```

## VIII. Using This Notebook

### What you'll need
Make sure your Python environment has these packages. Don't worry - the notebook will help you install anything that's missing.

**Core Dependencies:**
* `torch`
* `matplotlib`
* `numpy`
* `pandas`
* `seaborn`
* `networkx`
* `scipy`
* `datasets`
* `tiktoken`
* `tqdm`

**For the performance stuff (optional):**
* `nvidia-tensorrt` (for TensorRT optimization)
* `torch-tensorrt` (PyTorch-TensorRT integration)
* NVIDIA GPU with CUDA support (you'll need this for the optimization features)

### How to run it

1.  **Open the notebook:**
    * **Google Colab (I'd recommend this for GPU access):** Upload the `.ipynb` file or open it directly from GitHub
    * **Local setup:** Download and open with Jupyter Notebook, JupyterLab, or VS Code

2.  **Run the cells:** Just go through them in order. Here's what you'll find:
    * Setup and imports
    * The transformer model code
    * Helper functions for data and tracking
    * Visualization functions for attention analysis
    * The main menu where you choose what to do

3.  **Choose what you want to do:** Near the end, you'll get a menu with options:
    * **1. Train a new model:** (Heads up - this takes a while and needs a good GPU)
    * **2. Analyze an existing model:** Point it to a model you already have
    * **3. Do both:** Train first, then analyze
    Just type your choice when it asks.

4.  **Try the performance stuff (optional):** After you have a model, you can:
    * **Run the TensorRT cells:** These will optimize your model and show you the speed difference
    * **Use the one-click optimizer:** Just run `advanced_tensorrt_optimization_demo()`
    * **Profile your training:** Use `profile_training.py` to see where your training might be slow

### What you'll get

**The main stuff:**
* **Your trained model:** Saved in `results/` 
* **Pretty visualizations:** All the attention analysis charts and reports get saved too

**Performance optimization results:**
* **Optimized models:** The TensorRT versions get saved as `.ts` and `.pt` files
* **Speed comparisons:** You'll see how much faster (or not) things got
* **Profiling data:** If you use the profiler, you get a `.nsys-rep` file to analyze

**If you're using Colab:**
* The notebook will offer to download everything to your computer so you don't lose it

## IX. Customization & Further Exploration

### If you want to customize things

**For the analysis:**
* **Use your own model:** When you pick option 2, just point it to your model file (make sure it matches the architecture)
* **Try different sentences:** Edit `ALL_TEST_CATEGORIES` to analyze whatever text you want
* **Change the visuals:** Tweak the plotting functions to make the charts look how you like

**For the performance optimization:**
* **TensorRT tweaks:** You can change precision modes, input sizes, and how many benchmark runs to do
* **Profiling setup:** Adjust how long to profile for, or change the data loading settings for your hardware
* **Other stuff to try:** INT8 quantization if you want to go even faster (but it's more complex)

### What hardware you'll want
* **For basic stuff:** Any NVIDIA GPU that supports CUDA
* **For the optimization features:** Newer cards (RTX 30/40 series) will probably work better
* **Memory:** 8GB+ of GPU memory is helpful if you want to train larger models
