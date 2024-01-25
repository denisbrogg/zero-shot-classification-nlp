# %%
from sentence_transformers import SentenceTransformer
from zsl.components import zero_shot_predict
from zsl.plotting import visualize_embeddings


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
visualize_embeddings(X, labels_example_2, model, "PCA", 2)
# %%
visualize_embeddings(X, labels_example_2, model, "PCA", 3)
