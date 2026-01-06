## NLP for Computational Insights into Nutritional Impacts on Colorectal Cancer Care

### Main goal
- Improve colorectal cancer (CRC) prediction by combining dietary information with NLP and large language models.

### Key ideas
- Dietary data includes unstructured text (food descriptions) and structured data (lifestyle, nutrients).
- NLP preprocessing steps include:
  - Lowercasing
  - Stop-word removal
  - Punctuation removal
  - Term extraction
- NLP features are combined with structured data instead of being ignored.
- Uses an LLM-based framework to analyze dietary patterns.

### Important takeaway
- Treats nutrition as a modifiable risk factor, unlike genetics.
- Combining NLP-derived features with structured data significantly improves prediction accuracy.
- Shows how unstructured text can be useful in biomedical modeling.


## Personalized Causal Graph Reasoning for LLMs: A Case Study on Dietary Recommendations

### Main goal
- Improve LLM reasoning for dietary recommendations using causal graphs, not just correlations.

### Key ideas
- Standard LLMs rely on population-level patterns, which can give generic or risky advice.
- This paper builds personal causal graphs from individual data.
- LLM reasons over:
  - Nutrients
  - Biomarkers
  - Health outcomes
- Uses counterfactual evaluation to verify recommendations.

### Important takeaway
- Causal graphs guide LLM reasoning and make outputs:
  - More personalized
  - More interpretable
  - More causally valid
- Shows how LLMs + knowledge graphs can work together.


## MRAgent: An LLM-Based Automated Agent for Causal Knowledge Discovery via Mendelian Randomization

### Main goal
- Automate causal discovery in disease research using LLMs and Mendelian Randomization (MR).

### Key ideas
- MRAgent:
  - Reads biomedical literature
  - Extracts exposure–outcome pairs
  - Checks if MR studies already exist
  - Runs MR using GWAS data
- Uses LLMs for:
  - Process control
  - Decision-making
  - Report generation

### Important takeaway
- LLMs are used beyond text generation.
- Demonstrates how an agentic workflow can scale causal inference.
- Connects literature mining with genetic causal analysis.









