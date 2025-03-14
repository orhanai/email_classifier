from sklearn.datasets import fetch_20newsgroups

# Load the dataset (all categories)
newsgroups = fetch_20newsgroups(subset="all")
texts = newsgroups.data
labels = newsgroups.target
target_names = newsgroups.target_names

print(texts[0])
print(labels[0])
print(target_names[labels[0]])
