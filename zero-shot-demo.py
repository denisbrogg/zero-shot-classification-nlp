# %%
from sentence_transformers import SentenceTransformer
from zsl.components import zero_shot_predict


model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# %%

X = [
    "They won the world cup playing the best football ever seen",
    "These elections showed that the pirate party is still strong",
    "Leonardo Di Caprio lost the oscar last night because of his bad luck",
    "That scene was incredibly raw and the interpretation was incredible",
    "Ravens this year are going to win the super bowl",
    "Manchester City is a serious candidate to win this champions league",
]

labels_example_1 = ["positive", "negative"]
labels_example_2 = [
    "politics & elections",
    "sports & competitions",
    "entertainment & cinema",
    "finance & market",
]

# %%
zero_shot_predict(X, labels=labels_example_1, model=model)

# %%
zero_shot_predict(
    X,
    labels=labels_example_2,
    model=model,
)

# %%
from zsl.components import compute_embeddings
from zsl.plotting import (
    fit_transform_pca,
    visualize_embeddings_2d,
    visualize_embeddings_3d,
)

X_embeddings_pca_2d = fit_transform_pca(compute_embeddings(X, model))
label_embeddings_pca_2d = fit_transform_pca(compute_embeddings(labels_example_2, model))


visualize_embeddings_2d(
    x_pca=X_embeddings_pca_2d,
    x_pca_annotations=[" ".join(x.split(" ")[:2]) for x in X],
    labels_pca=label_embeddings_pca_2d,
    labels_pca_annotations=labels_example_2,
)

# %%
X_embeddings_pca_3d = fit_transform_pca(compute_embeddings(X, model), 3)
label_embeddings_pca_3d = fit_transform_pca(
    compute_embeddings(labels_example_2, model), 3
)


visualize_embeddings_3d(
    x_pca=X_embeddings_pca_3d,
    x_pca_annotations=[" ".join(x.split(" ")[:2]) for x in X],
    labels_pca=label_embeddings_pca_3d,
    labels_pca_annotations=labels_example_2,
)
# %%
