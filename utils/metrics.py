import torch

def calculate_mAP(query_codes, database_codes, query_labels, database_labels, max_samples=1000):
    """
    Calculate mean Average Precision (mAP) with sampling
    """
    num_query = min(query_labels.shape[0], max_samples)
    mean_AP = 0.0

    # Random sampling for faster evaluation
    indices = torch.randperm(query_labels.shape[0])[:num_query]
    query_codes = query_codes[indices]
    query_labels = query_labels[indices]

    # Compute all hamming distances at once
    hamming_dist = torch.cdist(query_codes.float(), database_codes.float(), p=1)
    
    # Compute mAP for each query in parallel
    for i in range(num_query):
        ind = torch.argsort(hamming_dist[i])
        query_label = query_labels[i]
        pos_num = torch.sum(database_labels == query_label)
        
        # Vectorized AP calculation
        retrieved = (database_labels[ind] == query_label)
        cumsum = torch.cumsum(retrieved.float(), dim=0)
        rank = torch.arange(1, len(retrieved) + 1, device=retrieved.device).float()
        precision = cumsum / rank
        mean_AP += torch.sum(precision * retrieved) / pos_num

    mean_AP /= num_query
    return mean_AP

def calculate_precision_at_k(query_codes, database_codes, query_labels, database_labels, k=1000, max_samples=1000):
    """
    Calculate Precision@K with sampling
    """
    num_query = min(query_labels.shape[0], max_samples)
    
    # Random sampling
    indices = torch.randperm(query_labels.shape[0])[:num_query]
    query_codes = query_codes[indices]
    query_labels = query_labels[indices]
    
    # Compute all hamming distances at once
    hamming_dist = torch.cdist(query_codes.float(), database_codes.float(), p=1)
    
    # Get top-k indices for all queries at once
    _, topk_indices = torch.topk(hamming_dist, k, dim=1, largest=False)
    
    # Compute precision for all queries
    retrieved_labels = database_labels[topk_indices]
    match = (retrieved_labels == query_labels.unsqueeze(1))
    precision = torch.sum(match.float(), dim=1) / k
    
    return torch.mean(precision)

def calculate_precision_recall_curve(query_codes, database_codes, query_labels, database_labels, max_samples=1000):
    """
    Calculate Precision-Recall curve points with sampling
    """
    num_query = min(query_labels.shape[0], max_samples)
    
    # Random sampling
    indices = torch.randperm(query_labels.shape[0])[:num_query]
    query_codes = query_codes[indices]
    query_labels = query_labels[indices]
    
    # Compute all hamming distances at once
    hamming_dist = torch.cdist(query_codes.float(), database_codes.float(), p=1)
    pr_points = []
    
    for i in range(num_query):
        # Sort by distance
        _, indices = torch.sort(hamming_dist[i])
        
        # Calculate precision and recall at each position
        retrieved_labels = database_labels[indices]
        relevant = (retrieved_labels == query_labels[i])
        
        # Cumulative sum of relevant items
        cum_relevant = torch.cumsum(relevant.float(), dim=0)
        precisions = cum_relevant / torch.arange(1, len(database_labels) + 1, device=cum_relevant.device)
        recalls = cum_relevant / torch.sum(database_labels == query_labels[i])
        
        pr_points.append((precisions, recalls))
    
    return pr_points 