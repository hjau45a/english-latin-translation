# English - Latin Translation

A minimalistic example how to fine-tune facebook's nllb-distilled translation model on an entierely "new" language.
Note: This project was done for fun, there is no deeper reason why I wanted to teach the model to be able to output Latin.

# Data

Data is taken from huggingface: grosenthal/latin_english_translation. This contains a lot of good quality examples for Latin - English sentence pairs.

# Training

Training is implemented in "seq2seq trainslation.ipynb". See comments and docstrings there.

# Results

Results are commented in "seq2seq translation eval.ipynb". What I observed:

1. The model appears to capture the essence of the language and produces translations that look very different from the target sentences, but according to an expert review on a random sample of fewer than 10 translations, they appear to be reasonable.
1. The standard metrics (BLEU, chrF, ter, METEOR) are quite bad.
1. Using a large LLM to check if the original input sentences can be reconstructed from the translations - comparing semantic similarity with an embedding model and cosine similarity - it appears that while the generated sentences are wildly different from the targets, they capture mostly the same semantic meaning. This suggests that the translation generally works, and produces useable translations.