# PART 2 HANDOFF README
- Trained on: title & text_no_dateline separately.
- Feature Fusion Method: hstack (scipy.sparse)
- Title TF-IDF params: {'min_df': 5, 'max_df': 0.9, 'ngram_range': (1, 2), 'sublinear_tf': True, 'stop_words': 'english', 'max_features': 5000}
- Body TF-IDF params: {'min_df': 5, 'max_df': 0.9, 'ngram_range': (1, 2), 'sublinear_tf': True, 'stop_words': 'english', 'max_features': 20000}
- RF Dimensionality Reduction: TruncatedSVD(n_components=100)
- Fine-tuning Applied: GridSearchCV
- Optimal Parameters Used for Refit:
  * MultinomialNB: alpha=0.1
  * SGDClassifier (SVM): loss='modified_huber', alpha=0.001
  * RandomForest: max_depth=15