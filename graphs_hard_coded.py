import matplotlib.pyplot as plt

# GRAPH 1: Model Performance

models = ["Baseline Top-1", "Baseline Top-3", "TF-IDF Top-1", "TF-IDF Top-3"]
scores = [36, 65, 58, 80]

plt.figure()
plt.bar(models, scores)
plt.title("Model Performance Comparison")
plt.xlabel("Model / Metric")
plt.ylabel("Accuracy / Hit Rate (%)")
plt.ylim(0, 100)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("graph1_model_performance.png")
plt.show()


# GRAPH 2: Similarity Scores


docs = ["Doc 1", "Doc 2", "Doc 3"]
similarity_scores = [0.608, 0.506, 0.489]  # from your burn example

plt.figure()
plt.bar(docs, similarity_scores)
plt.title("Top Retrieved Document Similarity Scores")
plt.xlabel("Retrieved Document")
plt.ylabel("Cosine Similarity Score")
plt.ylim(0, 0.7)
plt.tight_layout()
plt.savefig("graph2_similarity_scores.png")
plt.show()


# GRAPH 3: Category Distribution approximate

categories = ["Other", "Severe Bleeding", "CPR", "Burns", "Choking", "Other Categories"]
percents = [35, 15, 15, 10, 10, 15]

plt.figure()
plt.pie(percents, labels=categories, autopct='%1.0f%%')
plt.title("Category Distribution")
plt.tight_layout()
plt.savefig("graph3_category_distribution.png")
plt.show()