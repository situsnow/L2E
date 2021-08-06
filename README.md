# L2E
Code implementation for paper [Learning to Explain: Generating Stable Explanations Fast](https://aclanthology.org/2021.acl-long.415/) at ACL 2021, by Xuelin Situ, Ingrid Zukerman, Cecile Paris, Sameen Maruf and Reza Haffari.

# Requirements and Installation

- Python version >= 3.6.8
- PyTorch version >= 1.7.0
- HuggingFace transformers version >= 1.2.0
- [LIME](https://github.com/marcotcr/lime) >= 0.1.1.36
- [shap](https://github.com/slundberg/shap) == 0.29.3

# Experiments (steps to replicate the results from the paper)
1. **Collect explanations from different baselines** >> *preprocess.collect_base_explanations.py*
2. **Train L2E explainer (also refer to folder hyperparameters)** >> *learning2explain.py*
   
3. **Find neighbours for each test example (for stability evaluation)**:
   
   - For IMDB_R >> *evaluation.find_neighbours_imbdr.py*
   - For other datasets >> *evaluation.find_neighbours.py*
4. **Faithfulness evaluation**:

   - Prediction based >> *evaluation.compare_faithfulness_agreement.py*
   - Confidence based >> *evaluation.compare_faithfulness.py*
   - Prcision/Recall (for IMDB_R only) >> *evaluation.compare_imdbr_faithfulness.py*

5. **Stability evaluation**:
   
   - For IMDB_R >> *evaluation.compare_imdbr_stability.py*
   - For other datasets >> *evaluation.compare_stability.py*
    
6. **Efficiency evaluation** >> *compare_efficiency.py*



