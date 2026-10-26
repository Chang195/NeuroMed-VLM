import torch
def recall_at_k(labels,similarity,k):
    topk = similarity.topk(k=k,dim = 1).indices
    correct = labels.unsqueeze(1) == topk
    hits = correct.any(dim=1)
    return hits.float().mean().item()

def evaluate_retrieval(image_embeddings, text_embeddings):
    similarity = image_embeddings @ text_embeddings.T
    print("Similarity:", similarity.shape)
    labels = torch.arange(
        similarity.shape[0],
        device=similarity.device
    )

    results = {}

    for k in [1, 5, 10]:
        i2t = recall_at_k(labels, similarity, k)
        t2i = recall_at_k(labels, similarity.T, k)

        results[f"I2T_R@{k}"] = i2t
        results[f"T2I_R@{k}"] = t2i

    return results