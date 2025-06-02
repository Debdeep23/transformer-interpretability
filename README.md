# Pico-LLM Project Extension: Interpretability Analysis

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

## VII. Using This Notebook

### Dependencies
Ensure your Python environment (e.g., Google Colab) has the following packages. The notebook includes cells to help install missing ones.
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

### Running the Notebook

1.  **Open the Notebook:**
    * **Google Colab (Recommended for GPU access):** Upload the `.ipynb` file or open directly from GitHub (replace `github.com` with `colab.research.google.com/github/` in the URL).
    * **Local Jupyter Environment:** Download and open with Jupyter Notebook, JupyterLab, or VS Code.

2.  **Run the Cells:** Execute cells sequentially. The notebook is structured with:
    * Setup and Imports
    * Model Definitions (Transformer components)
    * Utility Functions (Data handling, Experiment Tracking)
    * Attention Visualization Functions
    * Main execution logic (controlled by user input)

3.  **Interact with the Menu:** Towards the end of the notebook, the `main()` function provides an interactive menu:
    * **1. Pre-train Base Transformer Model:** (Note: Pre-training is resource-intensive)
    * **2. Run Enhanced Interpretability Analysis:** Loads a pre-trained model (you'll be prompted for the path) for attention analysis.
    * **3. Run BOTH: Pre-train then Analyze:**
    Enter your choice in the cell's output field.

### Output
* **Model Checkpoints & Training Logs:** Saved in `results/` (default `pico_llm_interpretability_experiments/` if training).
* **Interpretability Visualizations & Reports:** Generated in `pico_llm_interpretability_experiments/attention_visualizations/<model_name>/`.
* **Colab Downloads:** If in Colab, the script prompts for automatic downloads of results to your local machine.

## VIII. Customization & Further Exploration

* **Model for Analysis:** To analyze a different pre-trained model, update the path when prompted by option '2' in the menu. Ensure the model architecture matches the one defined in the notebook (or adapt the definitions).
* **Test Sentences:** Modify `ALL_TEST_CATEGORIES` or the selection logic in `main()` to use different sentences for analysis.
* **Visualization Parameters:** Adjust parameters within the plotting functions in `PART 8` of the notebook for different visual outputs.
