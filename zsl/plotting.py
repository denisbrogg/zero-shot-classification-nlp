from typing import List, Literal
from sklearn.decomposition import PCA
from umap import UMAP
import matplotlib.pyplot as plt
from numpy.typing import NDArray

from zsl.components import Embedder, compute_embeddings


def fit_transform_projector(
    embeddings: NDArray,
    n_components: int = 2,
    projection: Literal["PCA", "UMAP"] = "UMAP",
) -> NDArray:
    projection_dict = {"PCA": PCA, "UMAP": UMAP}
    projector = projection_dict[projection](n_components=n_components)
    projections = projector.fit_transform(embeddings)
    return projections


#
# 2D
#


def scatter_and_annotate_2d(
    ax, vec_2d: NDArray, annotations: List = None, label_offset: tuple = (0, 0.02)
):
    ax.scatter(vec_2d[:, 0], vec_2d[:, 1])

    if annotations is not None:
        for i, label in enumerate(annotations):
            ax.annotate(
                label,
                (
                    vec_2d[i, 0] + label_offset[0],
                    vec_2d[i, 1] + label_offset[1],
                ),
            )


def visualize_embeddings_2d(
    x_pca,
    x_pca_annotations,
    labels_pca,
    labels_pca_annotations,
    title="2D PCA Visualization of Embeddings",
):
    fig, ax = plt.subplots(figsize=(8, 8))

    scatter_and_annotate_2d(ax, x_pca, x_pca_annotations)
    scatter_and_annotate_2d(ax, labels_pca, labels_pca_annotations)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.show()


#
# 3D
#


def scatter_and_annotate_3d(
    ax, vec_3d: NDArray, annotations: List = None, label_offset: tuple = (0, 0, 0.02)
):
    ax.scatter(vec_3d[:, 0], vec_3d[:, 1], vec_3d[:, 2])

    if annotations is not None:
        for i, label in enumerate(annotations):
            ax.text(
                vec_3d[i, 0] + label_offset[0],
                vec_3d[i, 1] + label_offset[1],
                vec_3d[i, 2] + label_offset[2],
                label,
                fontsize=8,
                color="black",
            )


def visualize_embeddings_3d(
    x_pca,
    x_pca_annotations,
    labels_pca,
    labels_pca_annotations,
    title,
):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")

    scatter_and_annotate_3d(ax, x_pca, x_pca_annotations)
    scatter_and_annotate_3d(ax, labels_pca, labels_pca_annotations)

    ax.set_title(title)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.show()


def visualize_embeddings(
    X,
    labels,
    model: Embedder,
    projection: Literal["PCA", "UMAP"] = "UMAP",
    dimensions: int = 2,
):
    if projection == "PCA":
        X_embeddings_projections = fit_transform_projector(
            compute_embeddings(X, model), n_components=dimensions, projection=projection
        )
        label_projections = fit_transform_projector(
            compute_embeddings(labels, model),
            n_components=dimensions,
            projection=projection,
        )
    elif projection == "UMAP":
        projections = fit_transform_projector(
            compute_embeddings(X + labels, model),
            n_components=dimensions,
            projection=projection,
        )
        X_embeddings_projections = projections[: len(X)]
        label_projections = projections[len(X) :]

    if dimensions == 2:
        visualize_embeddings_2d(
            x_pca=X_embeddings_projections,
            x_pca_annotations=[" ".join(x.split(" ")[:2]) for x in X],
            labels_pca=label_projections,
            labels_pca_annotations=labels,
            title=f"{dimensions}D {projection} Visualization of Embeddings",
        )
    elif dimensions == 3:
        visualize_embeddings_3d(
            x_pca=X_embeddings_projections,
            x_pca_annotations=[" ".join(x.split(" ")[:2]) for x in X],
            labels_pca=label_projections,
            labels_pca_annotations=labels,
            title=f"{dimensions}D {projection} Visualization of Embeddings",
        )
