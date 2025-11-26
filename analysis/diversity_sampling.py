"""
Diversity-Aware Sampling for Uncertain Tasks

When allocating limited external verification budget, we want to select tasks that are:
1. Uncertain (LLM verification may be unreliable)
2. Diverse (covering different types of problems, not redundant)

This module provides:
1. Task diversity metrics (structural, lexical, TF-IDF, sentence embeddings)
2. Diverse subset sampling algorithms (MMR, greedy, clustering, DPP)

Usage:
    python -m analysis.diversity_sampling --input_path results.json --budget 10 --uncertainty_threshold 0.5

    # With sentence embeddings
    python -m analysis.diversity_sampling --input_path results.json --budget 10 --similarity_method embedding

    # With DPP sampling
    python -m analysis.diversity_sampling --input_path results.json --budget 10 --diversity_method dpp
"""

import os
import sys
import json
import argparse
import math
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("Warning: sklearn not available, using basic diversity metrics")

# Sentence embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Warning: sentence-transformers not available, embedding similarity disabled")
    print("Install with: pip install sentence-transformers")


# ==============================================================================
# SENTENCE EMBEDDING
# ==============================================================================

class EmbeddingModel:
    """Wrapper for sentence embedding models."""

    _instance = None
    _model = None
    _model_name = None

    @classmethod
    def get_model(cls, model_name: str = "all-MiniLM-L6-v2"):
        """
        Get or create a singleton embedding model.

        Args:
            model_name: Name of the sentence-transformers model to use.
                       Options: "all-MiniLM-L6-v2" (fast), "all-mpnet-base-v2" (better quality),
                               "microsoft/codebert-base" (for code)
        """
        if not HAS_SENTENCE_TRANSFORMERS:
            return None

        if cls._model is None or cls._model_name != model_name:
            print(f"  Loading embedding model: {model_name}...")
            cls._model = SentenceTransformer(model_name)
            cls._model_name = model_name

        return cls._model

    @classmethod
    def encode(cls, texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> Optional[np.ndarray]:
        """
        Encode texts into embeddings.

        Args:
            texts: List of text strings to encode
            model_name: Embedding model to use

        Returns:
            np.ndarray of shape (n_texts, embedding_dim) or None if unavailable
        """
        model = cls.get_model(model_name)
        if model is None:
            return None

        embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embeddings


def compute_embedding_similarity(
    tasks: List[Dict],
    model_name: str = "all-MiniLM-L6-v2",
    text_field: str = "code_snippet",
) -> Optional[np.ndarray]:
    """
    Compute pairwise similarity using sentence embeddings.

    **Why sentence embeddings work for diversity:**
    Sentence embeddings capture semantic meaning of text in a dense vector space.
    Similar code/tasks will have similar embeddings (high cosine similarity).
    This captures semantic similarity beyond surface-level lexical overlap,
    identifying truly different problems even if they use similar syntax.

    Args:
        tasks: List of task dictionaries
        model_name: Sentence transformer model to use
        text_field: Which field to use for embedding ("code_snippet", "solver_prompt", etc.)

    Returns:
        n x n similarity matrix, or None if embeddings unavailable
    """
    if not HAS_SENTENCE_TRANSFORMERS:
        return None

    # Extract texts
    texts = []
    for task in tasks:
        text = task.get(text_field, "")
        if not text:
            # Fallback to combining multiple fields
            code = task.get("code_snippet", "")
            input_args = task.get("input_args", "")
            gold_output = task.get("gold_output", "")
            text = f"Code: {code}\nInput: {input_args}\nOutput: {gold_output}"
        texts.append(text)

    # Get embeddings
    embeddings = EmbeddingModel.encode(texts, model_name)

    if embeddings is None:
        return None

    # Compute cosine similarity
    # Normalize embeddings
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normalized = embeddings / (norms + 1e-10)

    # Cosine similarity matrix
    similarity_matrix = np.dot(normalized, normalized.T)

    # Clip to [0, 1] (cosine similarity can be slightly outside due to numerical errors)
    similarity_matrix = np.clip(similarity_matrix, 0, 1)

    return similarity_matrix


# ==============================================================================
# DETERMINANTAL POINT PROCESS (DPP)
# ==============================================================================

def sample_dpp(
    candidates: List[int],
    similarity_matrix: np.ndarray,
    budget: int,
    uncertainty_scores: List[float] = None,
    alpha: float = 1.0,
) -> List[int]:
    """
    Determinantal Point Process (DPP) sampling for diverse subset selection.

    **Why DPP works for diversity:**
    DPP is a probabilistic model that naturally captures repulsion/diversity.
    The probability of selecting a set S is proportional to det(L_S), where L_S
    is the submatrix of the kernel L indexed by S. Items that are similar
    (high kernel values) will have lower probability of being selected together
    because similar rows/columns make the determinant smaller.

    DPP elegantly balances:
    - Quality/relevance (diagonal of L, controlled by uncertainty scores)
    - Diversity (off-diagonal of L, controlled by similarity)

    The L-ensemble kernel is constructed as:
        L_ij = q_i * S_ij * q_j
    where q_i is the quality (uncertainty) score and S_ij is similarity.

    We use greedy MAP inference for tractable k-DPP sampling.

    Args:
        candidates: List of candidate task indices
        similarity_matrix: Pairwise similarity matrix (S)
        budget: Number of tasks to select (k)
        uncertainty_scores: Quality scores for each task (q)
        alpha: Scaling factor for quality scores (higher = more weight on quality)

    Returns:
        List of selected task indices
    """
    if len(candidates) <= budget:
        return candidates.copy()

    n = len(candidates)

    # Build quality scores
    if uncertainty_scores is not None:
        q = np.array([uncertainty_scores[i] for i in candidates])
        # Normalize and scale
        if q.max() > q.min():
            q = (q - q.min()) / (q.max() - q.min())
        else:
            q = np.ones(n)
        q = q * alpha + 0.1  # Add small constant for numerical stability
    else:
        q = np.ones(n)

    # Extract submatrix of similarities for candidates
    S = similarity_matrix[np.ix_(candidates, candidates)]

    # Build L-ensemble kernel: L_ij = q_i * S_ij * q_j
    # But we want diversity, so we use: L_ij = q_i * (1 - S_ij + delta_ij) * q_j
    # where delta_ij is 1 if i==j, 0 otherwise
    # This makes similar items have lower L values
    L = np.outer(q, q) * (np.eye(n) + 0.1 * (1 - S))

    # Greedy MAP inference for k-DPP
    # At each step, select the item that maximizes the marginal gain in log-det
    selected_local = []
    remaining = list(range(n))

    # Current selected kernel (starts empty)
    for _ in range(budget):
        if not remaining:
            break

        best_gain = -float('inf')
        best_idx = None

        for idx in remaining:
            # Compute marginal gain of adding idx
            if not selected_local:
                # First item: gain is just L[idx, idx]
                gain = np.log(L[idx, idx] + 1e-10)
            else:
                # Compute the conditional variance
                # Using Schur complement: L[idx,idx] - L[idx,S] @ inv(L[S,S]) @ L[S,idx]
                selected_arr = np.array(selected_local)
                L_SS = L[np.ix_(selected_arr, selected_arr)]
                L_iS = L[idx, selected_arr]

                try:
                    # Use solve instead of explicit inverse for numerical stability
                    L_SS_inv_L_Si = np.linalg.solve(L_SS + 1e-6 * np.eye(len(selected_arr)), L_iS)
                    conditional_var = L[idx, idx] - np.dot(L_iS, L_SS_inv_L_Si)
                    gain = np.log(max(conditional_var, 1e-10))
                except np.linalg.LinAlgError:
                    gain = np.log(L[idx, idx] + 1e-10)

            if gain > best_gain:
                best_gain = gain
                best_idx = idx

        if best_idx is not None:
            selected_local.append(best_idx)
            remaining.remove(best_idx)

    # Convert local indices back to global
    selected = [candidates[i] for i in selected_local]

    return selected


def sample_dpp_exact(
    candidates: List[int],
    similarity_matrix: np.ndarray,
    budget: int,
    uncertainty_scores: List[float] = None,
    alpha: float = 1.0,
    n_samples: int = 100,
) -> List[int]:
    """
    Exact k-DPP sampling using eigendecomposition.

    This method samples from the exact k-DPP distribution, then returns
    the sample with highest quality. More accurate but slower than greedy.

    Args:
        candidates: List of candidate task indices
        similarity_matrix: Pairwise similarity matrix
        budget: Number of tasks to select
        uncertainty_scores: Quality scores
        alpha: Quality weight
        n_samples: Number of DPP samples to draw

    Returns:
        Selected task indices
    """
    if len(candidates) <= budget:
        return candidates.copy()

    n = len(candidates)

    # Build quality scores
    if uncertainty_scores is not None:
        q = np.array([uncertainty_scores[i] for i in candidates])
        if q.max() > q.min():
            q = (q - q.min()) / (q.max() - q.min())
        else:
            q = np.ones(n)
        q = q * alpha + 0.1
    else:
        q = np.ones(n)

    # Extract similarity submatrix
    S = similarity_matrix[np.ix_(candidates, candidates)]

    # Build kernel: L = diag(q) @ (I + 0.1*(1-S)) @ diag(q)
    L = np.outer(q, q) * (np.eye(n) + 0.1 * (1 - S))

    # Eigendecomposition
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(L)
        eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative
    except np.linalg.LinAlgError:
        # Fallback to greedy
        return sample_dpp(candidates, similarity_matrix, budget, uncertainty_scores, alpha)

    # Sample from k-DPP
    best_sample = None
    best_quality = -float('inf')

    for _ in range(n_samples):
        # Sample k eigenvalues with probability proportional to lambda_i / (1 + lambda_i)
        probs = eigenvalues / (1 + eigenvalues + 1e-10)
        probs = probs / (probs.sum() + 1e-10)

        # Sample indices
        try:
            selected_eigenvec_indices = np.random.choice(
                n, size=min(budget, n), replace=False, p=probs
            )
        except ValueError:
            selected_eigenvec_indices = np.random.choice(n, size=min(budget, n), replace=False)

        # Project to get item probabilities
        V = eigenvectors[:, selected_eigenvec_indices]

        # Greedy selection from the projected space
        selected_local = []
        remaining = list(range(n))

        for _ in range(budget):
            if not remaining or V.shape[1] == 0:
                break

            # Compute squared norms
            norms = np.sum(V[remaining] ** 2, axis=1)
            if norms.sum() < 1e-10:
                break

            # Sample proportional to norms
            probs_item = norms / (norms.sum() + 1e-10)
            try:
                chosen_local = np.random.choice(len(remaining), p=probs_item)
            except ValueError:
                chosen_local = np.random.randint(len(remaining))

            chosen = remaining[chosen_local]
            selected_local.append(chosen)
            remaining.remove(chosen)

            # Update V by projecting out the selected item
            if len(remaining) > 0 and V.shape[1] > 0:
                v_chosen = V[chosen]
                norm_v = np.dot(v_chosen, v_chosen)
                if norm_v > 1e-10:
                    V = V - np.outer(V @ v_chosen, v_chosen) / norm_v

        # Compute quality of this sample
        if selected_local:
            sample_quality = sum(q[i] for i in selected_local)
            if sample_quality > best_quality:
                best_quality = sample_quality
                best_sample = selected_local

    if best_sample is None:
        return sample_dpp(candidates, similarity_matrix, budget, uncertainty_scores, alpha)

    return [candidates[i] for i in best_sample]


# ==============================================================================
# TASK FEATURE EXTRACTION
# ==============================================================================

def extract_task_features(task: Dict) -> Dict:
    """
    Extract features from a task for diversity computation.

    Features include:
    - Code structure (AST-like features)
    - Lexical features (tokens, keywords)
    - Problem type
    - Input/output characteristics
    """
    features = {
        "task_id": task.get("task_id", ""),
        "problem_type": task.get("problem_type", ""),
    }

    code = task.get("code_snippet", "")
    if code:
        # Lexical features
        features["code_length"] = len(code)
        features["n_lines"] = code.count('\n') + 1
        features["n_functions"] = code.count('def ')
        features["n_loops"] = code.count('for ') + code.count('while ')
        features["n_conditionals"] = code.count('if ') + code.count('elif ') + code.count('else:')
        features["n_returns"] = code.count('return ')

        # Keyword features
        keywords = ['list', 'dict', 'set', 'tuple', 'str', 'int', 'float',
                    'len', 'range', 'enumerate', 'zip', 'map', 'filter',
                    'sum', 'max', 'min', 'sorted', 'reversed']
        for kw in keywords:
            features[f"has_{kw}"] = 1 if kw in code else 0

        # Code text for TF-IDF
        features["code_text"] = code

    # Input/output characteristics
    input_args = task.get("input_args", "")
    if input_args:
        features["input_length"] = len(str(input_args))
        features["input_text"] = str(input_args)

    gold_output = task.get("gold_output", "")
    if gold_output:
        features["output_length"] = len(str(gold_output))
        features["output_text"] = str(gold_output)

    return features


def compute_structural_similarity(features1: Dict, features2: Dict) -> float:
    """
    Compute structural similarity between two tasks based on code features.

    Returns similarity score in [0, 1] where 1 means identical structure.
    """
    # Numerical features to compare
    num_features = [
        "code_length", "n_lines", "n_functions", "n_loops",
        "n_conditionals", "n_returns", "input_length", "output_length"
    ]

    # Binary features (keywords)
    bin_features = [k for k in features1.keys() if k.startswith("has_")]

    # Compute numerical similarity (normalized difference)
    num_sim = 0.0
    num_count = 0
    for feat in num_features:
        v1 = features1.get(feat, 0)
        v2 = features2.get(feat, 0)
        if v1 > 0 or v2 > 0:
            sim = 1 - abs(v1 - v2) / (max(v1, v2) + 1e-10)
            num_sim += sim
            num_count += 1

    # Compute binary feature similarity (Jaccard)
    set1 = {f for f in bin_features if features1.get(f, 0) == 1}
    set2 = {f for f in bin_features if features2.get(f, 0) == 1}
    if set1 or set2:
        jaccard = len(set1 & set2) / len(set1 | set2)
    else:
        jaccard = 1.0

    # Problem type match
    type_match = 1.0 if features1.get("problem_type") == features2.get("problem_type") else 0.0

    # Weighted combination
    if num_count > 0:
        structural_sim = 0.4 * (num_sim / num_count) + 0.4 * jaccard + 0.2 * type_match
    else:
        structural_sim = 0.6 * jaccard + 0.4 * type_match

    return structural_sim


def compute_lexical_similarity(text1: str, text2: str) -> float:
    """
    Compute lexical similarity using character n-grams.
    """
    if not text1 or not text2:
        return 0.0

    # Character 3-grams
    def get_ngrams(text: str, n: int = 3) -> Set[str]:
        return set(text[i:i+n] for i in range(len(text) - n + 1))

    ngrams1 = get_ngrams(text1)
    ngrams2 = get_ngrams(text2)

    if not ngrams1 or not ngrams2:
        return 0.0

    # Jaccard similarity
    intersection = len(ngrams1 & ngrams2)
    union = len(ngrams1 | ngrams2)

    return intersection / union if union > 0 else 0.0


# ==============================================================================
# DIVERSITY METRICS
# ==============================================================================

def compute_pairwise_similarities(
    tasks: List[Dict],
    method: str = "combined",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> np.ndarray:
    """
    Compute pairwise similarity matrix for tasks.

    Args:
        tasks: List of task dictionaries
        method: "structural", "lexical", "tfidf", "embedding", or "combined"
        embedding_model: Model name for sentence embeddings

    Returns:
        n x n similarity matrix
    """
    n = len(tasks)
    similarities = np.zeros((n, n))

    # Embedding-based similarity
    if method == "embedding":
        if HAS_SENTENCE_TRANSFORMERS:
            sim = compute_embedding_similarity(tasks, embedding_model)
            if sim is not None:
                return sim
        print("  Warning: Embedding similarity unavailable, falling back to combined")
        method = "combined"

    # Extract features
    features_list = [extract_task_features(t) for t in tasks]

    if method == "tfidf" and HAS_SKLEARN:
        # TF-IDF on code text
        code_texts = [f.get("code_text", "") for f in features_list]
        if all(code_texts):
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
            try:
                tfidf_matrix = vectorizer.fit_transform(code_texts)
                similarities = cosine_similarity(tfidf_matrix)
                return similarities
            except Exception:
                method = "combined"  # Fallback

    if method in ["structural", "lexical", "combined"]:
        for i in range(n):
            similarities[i, i] = 1.0
            for j in range(i + 1, n):
                if method == "structural":
                    sim = compute_structural_similarity(features_list[i], features_list[j])
                elif method == "lexical":
                    code1 = features_list[i].get("code_text", "")
                    code2 = features_list[j].get("code_text", "")
                    sim = compute_lexical_similarity(code1, code2)
                else:  # combined
                    struct_sim = compute_structural_similarity(features_list[i], features_list[j])
                    code1 = features_list[i].get("code_text", "")
                    code2 = features_list[j].get("code_text", "")
                    lex_sim = compute_lexical_similarity(code1, code2)
                    sim = 0.5 * struct_sim + 0.5 * lex_sim

                similarities[i, j] = sim
                similarities[j, i] = sim

    return similarities


def compute_set_diversity(
    selected_indices: List[int],
    similarity_matrix: np.ndarray
) -> float:
    """
    Compute diversity of a selected set of tasks.

    Diversity = 1 - average pairwise similarity

    Higher values mean more diverse selection.
    """
    if len(selected_indices) < 2:
        return 1.0

    total_sim = 0.0
    n_pairs = 0

    for i, idx1 in enumerate(selected_indices):
        for idx2 in selected_indices[i+1:]:
            total_sim += similarity_matrix[idx1, idx2]
            n_pairs += 1

    avg_similarity = total_sim / n_pairs if n_pairs > 0 else 0.0
    diversity = 1.0 - avg_similarity

    return diversity


# ==============================================================================
# DIVERSE SUBSET SAMPLING ALGORITHMS
# ==============================================================================

def sample_greedy_diverse(
    candidates: List[int],
    similarity_matrix: np.ndarray,
    budget: int,
    uncertainty_scores: List[float] = None,
    lambda_diversity: float = 0.5,
) -> List[int]:
    """
    Greedy diversity sampling with optional uncertainty weighting.

    At each step, select the candidate that maximizes:
        score = lambda * uncertainty + (1 - lambda) * min_distance_to_selected

    Args:
        candidates: List of candidate task indices
        similarity_matrix: Pairwise similarity matrix
        budget: Number of tasks to select
        uncertainty_scores: Uncertainty score for each candidate (higher = more uncertain)
        lambda_diversity: Weight for diversity vs uncertainty (0 = only diversity, 1 = only uncertainty)

    Returns:
        List of selected task indices
    """
    if len(candidates) <= budget:
        return candidates.copy()

    # Normalize uncertainty scores if provided
    if uncertainty_scores is not None:
        u_scores = np.array([uncertainty_scores[i] for i in candidates])
        if u_scores.max() > u_scores.min():
            u_scores = (u_scores - u_scores.min()) / (u_scores.max() - u_scores.min())
        else:
            u_scores = np.ones(len(candidates))
    else:
        u_scores = np.ones(len(candidates))
        lambda_diversity = 0.0  # Only use diversity if no uncertainty scores

    selected = []
    remaining = set(candidates)

    # Select first item (highest uncertainty)
    first_idx = candidates[np.argmax(u_scores)]
    selected.append(first_idx)
    remaining.remove(first_idx)

    # Greedy selection
    while len(selected) < budget and remaining:
        best_score = -float('inf')
        best_idx = None

        for idx in remaining:
            # Diversity score: minimum distance to any selected item
            # Distance = 1 - similarity
            min_distance = min(1 - similarity_matrix[idx, s] for s in selected)

            # Combined score
            cand_idx_in_list = candidates.index(idx)
            uncertainty = u_scores[cand_idx_in_list]
            score = lambda_diversity * uncertainty + (1 - lambda_diversity) * min_distance

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return selected


def sample_mmr(
    candidates: List[int],
    similarity_matrix: np.ndarray,
    budget: int,
    uncertainty_scores: List[float] = None,
    lambda_mmr: float = 0.5,
) -> List[int]:
    """
    Maximal Marginal Relevance (MMR) sampling.

    MMR balances relevance (uncertainty) and diversity by selecting items that are
    both relevant and different from already selected items.

    score = lambda * relevance - (1 - lambda) * max_similarity_to_selected

    Args:
        candidates: List of candidate task indices
        similarity_matrix: Pairwise similarity matrix
        budget: Number of tasks to select
        uncertainty_scores: Relevance score for each candidate
        lambda_mmr: Trade-off parameter (higher = more weight on relevance)

    Returns:
        List of selected task indices
    """
    if len(candidates) <= budget:
        return candidates.copy()

    # Normalize uncertainty scores
    if uncertainty_scores is not None:
        relevance = np.array([uncertainty_scores[i] for i in candidates])
        if relevance.max() > relevance.min():
            relevance = (relevance - relevance.min()) / (relevance.max() - relevance.min())
        else:
            relevance = np.ones(len(candidates))
    else:
        relevance = np.ones(len(candidates))

    selected = []
    remaining = list(range(len(candidates)))

    # Select first item (highest relevance)
    first_local_idx = np.argmax(relevance)
    selected.append(candidates[first_local_idx])
    remaining.remove(first_local_idx)

    # MMR selection
    while len(selected) < budget and remaining:
        best_score = -float('inf')
        best_local_idx = None

        for local_idx in remaining:
            global_idx = candidates[local_idx]

            # Relevance term
            rel = relevance[local_idx]

            # Redundancy term: max similarity to any selected item
            max_sim = max(similarity_matrix[global_idx, s] for s in selected)

            # MMR score
            mmr_score = lambda_mmr * rel - (1 - lambda_mmr) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_local_idx = local_idx

        if best_local_idx is not None:
            selected.append(candidates[best_local_idx])
            remaining.remove(best_local_idx)

    return selected


def sample_clustering(
    candidates: List[int],
    similarity_matrix: np.ndarray,
    budget: int,
    uncertainty_scores: List[float] = None,
) -> List[int]:
    """
    Clustering-based diverse sampling.

    1. Cluster candidates into k clusters
    2. From each cluster, select the most uncertain item

    Args:
        candidates: List of candidate task indices
        similarity_matrix: Pairwise similarity matrix
        budget: Number of tasks to select
        uncertainty_scores: Uncertainty score for each candidate

    Returns:
        List of selected task indices
    """
    if len(candidates) <= budget:
        return candidates.copy()

    if not HAS_SKLEARN:
        # Fallback to greedy
        return sample_greedy_diverse(candidates, similarity_matrix, budget, uncertainty_scores)

    # Convert similarity to distance
    distance_matrix = 1 - similarity_matrix[np.ix_(candidates, candidates)]

    # K-means clustering on distance features
    n_clusters = min(budget, len(candidates))

    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        # Use distance matrix rows as features
        cluster_labels = kmeans.fit_predict(distance_matrix)
    except Exception:
        return sample_greedy_diverse(candidates, similarity_matrix, budget, uncertainty_scores)

    # From each cluster, select the most uncertain item
    selected = []
    cluster_to_candidates = defaultdict(list)

    for local_idx, cluster in enumerate(cluster_labels):
        cluster_to_candidates[cluster].append(local_idx)

    if uncertainty_scores is not None:
        u_scores = [uncertainty_scores[candidates[i]] for i in range(len(candidates))]
    else:
        u_scores = [1.0] * len(candidates)

    for cluster in range(n_clusters):
        cluster_members = cluster_to_candidates[cluster]
        if cluster_members:
            # Select most uncertain in cluster
            best_local_idx = max(cluster_members, key=lambda i: u_scores[i])
            selected.append(candidates[best_local_idx])

    # If budget not filled, add more from remaining
    remaining = [candidates[i] for i in range(len(candidates)) if candidates[i] not in selected]
    remaining.sort(key=lambda idx: uncertainty_scores[idx] if uncertainty_scores else 0, reverse=True)

    while len(selected) < budget and remaining:
        selected.append(remaining.pop(0))

    return selected[:budget]


# ==============================================================================
# HIGH-LEVEL API
# ==============================================================================

def select_diverse_uncertain_tasks(
    tasks: List[Dict],
    budget: int,
    uncertainty_metric: str = "sc_entropy",
    uncertainty_threshold: float = None,
    diversity_method: str = "mmr",
    lambda_tradeoff: float = 0.5,
    similarity_method: str = "combined",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> Tuple[List[int], Dict]:
    """
    Select a diverse subset of uncertain tasks for external verification.

    Args:
        tasks: List of task dictionaries with uncertainty_metrics
        budget: Number of tasks to select
        uncertainty_metric: Which UQ metric to use for filtering/ranking
        uncertainty_threshold: Only consider tasks above this uncertainty (None = use all)
        diversity_method: "greedy", "mmr", "clustering", "dpp", or "dpp_exact"
        lambda_tradeoff: Balance between uncertainty and diversity (0-1)
        similarity_method: "structural", "lexical", "tfidf", "embedding", or "combined"
        embedding_model: Model for sentence embeddings (if similarity_method="embedding")

    Returns:
        (selected_indices, stats): Selected task indices and selection statistics
    """
    # Get uncertainty scores
    uncertainty_scores = []
    valid_indices = []

    for i, task in enumerate(tasks):
        uq = task.get("uncertainty_metrics", {})
        score = uq.get(uncertainty_metric)

        if score is not None and not (isinstance(score, float) and math.isnan(score)):
            uncertainty_scores.append(score)
            valid_indices.append(i)
        else:
            uncertainty_scores.append(0.0)

    if not valid_indices:
        return [], {"error": "no_valid_tasks"}

    # Filter by threshold if specified
    if uncertainty_threshold is not None:
        # For metrics where higher = more uncertain
        candidates = [i for i in valid_indices if uncertainty_scores[i] >= uncertainty_threshold]
    else:
        candidates = valid_indices

    if not candidates:
        return [], {"error": "no_candidates_above_threshold"}

    stats = {
        "n_total_tasks": len(tasks),
        "n_valid_tasks": len(valid_indices),
        "n_candidates": len(candidates),
        "budget": budget,
        "uncertainty_metric": uncertainty_metric,
        "uncertainty_threshold": uncertainty_threshold,
        "diversity_method": diversity_method,
        "similarity_method": similarity_method,
        "lambda_tradeoff": lambda_tradeoff,
    }

    # If fewer candidates than budget, return all
    if len(candidates) <= budget:
        stats["n_selected"] = len(candidates)
        stats["diversity_score"] = 1.0  # Trivial case
        return candidates, stats

    # Compute similarity matrix
    print(f"  Computing {similarity_method} similarities...")
    similarity_matrix = compute_pairwise_similarities(
        tasks, method=similarity_method, embedding_model=embedding_model
    )

    # Sample diverse subset
    print(f"  Running {diversity_method} sampling...")
    if diversity_method == "greedy":
        selected = sample_greedy_diverse(
            candidates, similarity_matrix, budget,
            uncertainty_scores, lambda_tradeoff
        )
    elif diversity_method == "mmr":
        selected = sample_mmr(
            candidates, similarity_matrix, budget,
            uncertainty_scores, lambda_tradeoff
        )
    elif diversity_method == "clustering":
        selected = sample_clustering(
            candidates, similarity_matrix, budget,
            uncertainty_scores
        )
    elif diversity_method == "dpp":
        selected = sample_dpp(
            candidates, similarity_matrix, budget,
            uncertainty_scores, alpha=lambda_tradeoff
        )
    elif diversity_method == "dpp_exact":
        selected = sample_dpp_exact(
            candidates, similarity_matrix, budget,
            uncertainty_scores, alpha=lambda_tradeoff
        )
    else:
        raise ValueError(f"Unknown diversity method: {diversity_method}")

    # Compute statistics
    stats["n_selected"] = len(selected)
    stats["diversity_score"] = compute_set_diversity(selected, similarity_matrix)

    # Compare to random selection baseline
    random_samples = []
    for _ in range(100):
        random_sel = np.random.choice(candidates, size=min(budget, len(candidates)), replace=False)
        random_samples.append(compute_set_diversity(random_sel.tolist(), similarity_matrix))
    stats["random_baseline_diversity"] = np.mean(random_samples)
    stats["diversity_improvement"] = stats["diversity_score"] / stats["random_baseline_diversity"] if stats["random_baseline_diversity"] > 0 else 1.0

    # Compare to uncertainty-only selection baseline
    top_uncertain = sorted(candidates, key=lambda i: uncertainty_scores[i], reverse=True)[:budget]
    stats["uncertainty_only_diversity"] = compute_set_diversity(top_uncertain, similarity_matrix)

    # Selected tasks' uncertainty scores
    selected_uncertainties = [uncertainty_scores[i] for i in selected]
    stats["mean_selected_uncertainty"] = np.mean(selected_uncertainties)
    stats["min_selected_uncertainty"] = np.min(selected_uncertainties)
    stats["max_selected_uncertainty"] = np.max(selected_uncertainties)

    return selected, stats


def print_selection_summary(
    tasks: List[Dict],
    selected_indices: List[int],
    stats: Dict,
):
    """Print summary of the diverse selection."""
    print("\n" + "=" * 80)
    print("DIVERSE UNCERTAIN TASK SELECTION SUMMARY")
    print("=" * 80)

    print("\nSelection Parameters:")
    print(f"  Uncertainty metric: {stats.get('uncertainty_metric', 'N/A')}")
    print(f"  Uncertainty threshold: {stats.get('uncertainty_threshold', 'None')}")
    print(f"  Diversity method: {stats.get('diversity_method', 'N/A')}")
    print(f"  Similarity method: {stats.get('similarity_method', 'N/A')}")
    print(f"  Lambda (uncertainty vs diversity): {stats.get('lambda_tradeoff', 'N/A')}")

    print("\nSelection Results:")
    print(f"  Total tasks: {stats.get('n_total_tasks', 0)}")
    print(f"  Valid tasks (with UQ): {stats.get('n_valid_tasks', 0)}")
    print(f"  Candidates (above threshold): {stats.get('n_candidates', 0)}")
    print(f"  Budget: {stats.get('budget', 0)}")
    print(f"  Selected: {stats.get('n_selected', 0)}")

    print("\nDiversity Analysis:")
    print(f"  Selected set diversity: {stats.get('diversity_score', 0):.4f}")
    print(f"  Random baseline diversity: {stats.get('random_baseline_diversity', 0):.4f}")
    print(f"  Uncertainty-only diversity: {stats.get('uncertainty_only_diversity', 0):.4f}")
    print(f"  Improvement over random: {stats.get('diversity_improvement', 1):.2f}x")

    print("\nUncertainty of Selected Tasks:")
    print(f"  Mean: {stats.get('mean_selected_uncertainty', 0):.4f}")
    print(f"  Min: {stats.get('min_selected_uncertainty', 0):.4f}")
    print(f"  Max: {stats.get('max_selected_uncertainty', 0):.4f}")

    if selected_indices:
        print("\nSelected Task IDs:")
        for i, idx in enumerate(selected_indices[:20]):  # Show first 20
            task = tasks[idx]
            task_id = task.get("task_id", f"task_{idx}")
            uq = task.get("uncertainty_metrics", {})
            uncertainty = uq.get(stats.get("uncertainty_metric", ""), 0)
            print(f"  {i+1}. {task_id} (uncertainty={uncertainty:.4f})")
        if len(selected_indices) > 20:
            print(f"  ... and {len(selected_indices) - 20} more")

    print()


def main():
    parser = argparse.ArgumentParser(description="Select diverse uncertain tasks for external verification")
    parser.add_argument("--input_path", type=str, required=True, help="Path to results JSON")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save selected tasks")
    parser.add_argument("--budget", type=int, default=10, help="Number of tasks to select")
    parser.add_argument("--uncertainty_metric", type=str, default="sc_entropy",
                        help="UQ metric to use (sc_entropy, agreement_rate, perplexity, etc.)")
    parser.add_argument("--uncertainty_threshold", type=float, default=None,
                        help="Only consider tasks above this uncertainty threshold")
    parser.add_argument("--diversity_method", type=str, default="mmr",
                        choices=["greedy", "mmr", "clustering", "dpp", "dpp_exact"],
                        help="Diversity sampling method")
    parser.add_argument("--lambda_tradeoff", type=float, default=0.5,
                        help="Balance between uncertainty (1.0) and diversity (0.0)")
    parser.add_argument("--similarity_method", type=str, default="combined",
                        choices=["structural", "lexical", "tfidf", "embedding", "combined"],
                        help="Method for computing task similarity")
    parser.add_argument("--embedding_model", type=str, default="all-MiniLM-L6-v2",
                        help="Sentence transformer model for embedding similarity")

    args = parser.parse_args()

    print("=" * 80)
    print("DIVERSE UNCERTAIN TASK SELECTION")
    print("=" * 80)

    # Load tasks
    print(f"\nLoading tasks from: {args.input_path}")
    with open(args.input_path, 'r') as f:
        tasks = json.load(f)
    print(f"  Loaded {len(tasks)} tasks")

    # Select diverse uncertain tasks
    selected_indices, stats = select_diverse_uncertain_tasks(
        tasks=tasks,
        budget=args.budget,
        uncertainty_metric=args.uncertainty_metric,
        uncertainty_threshold=args.uncertainty_threshold,
        diversity_method=args.diversity_method,
        lambda_tradeoff=args.lambda_tradeoff,
        similarity_method=args.similarity_method,
        embedding_model=args.embedding_model,
    )

    # Print summary
    print_selection_summary(tasks, selected_indices, stats)

    # Save selected tasks if output path specified
    if args.output_path and selected_indices:
        selected_tasks = [tasks[i] for i in selected_indices]
        output_data = {
            "selection_stats": stats,
            "selected_task_indices": selected_indices,
            "selected_tasks": selected_tasks,
        }
        with open(args.output_path, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"Selected tasks saved to: {args.output_path}")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return selected_indices, stats


if __name__ == "__main__":
    main()
