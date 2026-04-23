import torch
from transformers import pipeline
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report,
                             confusion_matrix)
import time

print("=" * 55)
print("  Movie Review Sentiment Analyzer")
print("  Built with HuggingFace Transformers")
print("=" * 55)

# Step 1: Load real IMDB dataset
print("\n[1/5] Loading IMDB dataset...")
dataset = load_dataset("imdb")

# Use 100 test samples to keep it fast
test_samples = dataset["test"].shuffle(seed=42).select(range(100))
reviews  = test_samples["text"]
labels   = test_samples["label"]  # 0=negative 1=positive

print(f"Loaded {len(reviews)} movie reviews for evaluation")
print(f"Sample review (first 80 chars):")
print(f'"{reviews[0][:80]}..."')

# Step 2: Load pretrained model 
print("\n[2/5] Loading pretrained sentiment model...")
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    truncation=True,
    max_length=512
)
print("Model loaded!")

# Step 3: Run predictions 
print("\n[3/5] Analyzing 100 reviews...")
start_time = time.time()

predictions = []
for i, review in enumerate(reviews):
    result = sentiment_pipeline(review)[0]
    pred = 1 if result["label"] == "POSITIVE" else 0
    predictions.append(pred)
    if (i + 1) % 25 == 0:
        print(f"  Processed {i+1}/100 reviews...")

elapsed = time.time() - start_time
print(f"Done! Took {elapsed:.1f} seconds")

#  Step 4: Evaluate 
print("\n[4/5] Calculating metrics...")
accuracy  = accuracy_score(labels, predictions)
f1        = f1_score(labels, predictions)
cm        = confusion_matrix(labels, predictions)
report    = classification_report(
    labels, predictions,
    target_names=["Negative", "Positive"]
)

print("\n" + "=" * 55)
print("  EVALUATION REPORT")
print("=" * 55)
print(f"  Accuracy:  {accuracy:.1%}")
print(f"  F1 Score:  {f1:.1%}")
print("\n  Confusion Matrix:")
print(f"  True Neg:  {cm[0][0]}  False Pos: {cm[0][1]}")
print(f"  False Neg: {cm[1][0]}  True Pos:  {cm[1][1]}")
print("\n  Detailed Report:")
print(report)

# ── Step 5: Test your own reviews ─────────────────────────
print("=" * 55)
print("  [5/5] YOUR OWN REVIEWS")
print("=" * 55)

# Add your own movie reviews here!
my_reviews = [
    "This film was a masterpiece. Every scene was perfect.",
    "Terrible movie. Boring plot and bad acting.",
    "I had no idea what was happening for most of the film.",
    "An unknown gem — nobody talks about this but it deserves more.",
    "The special effects were great but the story made no sense.",
]

print("\nAnalyzing your custom reviews:\n")
for review in my_reviews:
    result = sentiment_pipeline(review)[0]
    label  = result["label"]
    conf   = result["score"]
    emoji  = "😊" if label == "POSITIVE" else "😞"
    print(f"{emoji} {label} ({conf:.0%})")
    print(f'   "{review}"')
    print()

# ── Bonus: Find where model struggles ─────────────────────
print("=" * 55)
print("  BONUS: Reviews the model got WRONG")
print("=" * 55)
wrong = 0
for i, (review, label, pred) in enumerate(
        zip(reviews, labels, predictions)):
    if label != pred and wrong < 3:
        actual    = "POSITIVE" if label == 1 else "NEGATIVE"
        predicted = "POSITIVE" if pred  == 1 else "NEGATIVE"
        print(f"\nReview #{i+1}:")
        print(f'  Text: "{review[:100]}..."')
        print(f"  Actual: {actual} | Predicted: {predicted}")
        wrong += 1
