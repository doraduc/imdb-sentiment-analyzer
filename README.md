# Movie Review Sentiment Analyzer 

A complete NLP pipeline that analyzes whether movie reviews 
are positive or negative using a pretrained transformer model.
Achieves 90% accuracy on 100 real IMDB reviews.

## Results
- Accuracy: 90.0%
- F1 Score: 88.9%
- True Positives: 40 · True Negatives: 50
- False Positives: 3 · False Negatives: 7
- Dataset: IMDB (50,000 real movie reviews)
- Model: DistilBERT fine-tuned on SST-2

## What I learned
- Why transfer learning matters — instead of training from
  scratch on millions of examples, I downloaded a pretrained
  model and got 90% accuracy in minutes
- The difference between plot description and reviewer opinion
  is one of the hardest problems in sentiment analysis —
  the model sometimes mistakes "the ship sank dramatically"
  (plot) for negative sentiment (opinion)
- What a confusion matrix really means in a real project —
  7 false negatives means 7 positive reviews wrongly flagged
  as negative, which in a real app would frustrate users
- How to do proper ML evaluation beyond just accuracy

## How to run
```bash
git clone https://github.com/doraduc/imdb-sentiment-analyzer
cd imdb-sentiment-analyzer
pip install -r requirements.txt
python sentiment_analyzer.py
```

## Pipeline
```
Raw review text
  → DistilBERT tokenizer (text → numbers)
  → Pretrained transformer (12 attention layers)
  → Sentiment classification (POSITIVE / NEGATIVE)
  → Confidence score (0-100%)
  → Evaluation report
```

## Real world problem identified
Current AI sentiment models struggle with reviews of obscure 
or less famous films — the model has limited knowledge of 
niche movies and can misread ambiguous language. Next steps 
would be fine-tuning on domain-specific movie review data 
to improve performance on lesser-known films.

## Tech stack
Python · HuggingFace Transformers · DistilBERT · 
datasets · scikit-learn · PyTorch

## Key insight
The model failed most on reviews where the writer described 
dark or dramatic plot events — it confused "this movie is 
about tragedy" with "I didn't like this movie." This is an 
active research problem in NLP called opinion vs description 
separation.
