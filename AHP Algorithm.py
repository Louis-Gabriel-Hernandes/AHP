import numpy as np

def create_pairwise_comparison_matrix(criteria, comparisons):
    n = len(criteria)
    matrix = np.ones((n, n))
    
    for (i, j), value in comparisons.items():
        matrix[i][j] = value
        matrix[j][i] = 1 / value
    
    return matrix

def normalize_matrix(matrix):
    col_sum = matrix.sum(axis=0)
    normalized_matrix = matrix / col_sum
    return normalized_matrix

def calculate_weights(normalized_matrix):
    weights = normalized_matrix.mean(axis=1)
    return weights

def calculate_consistency_ratio(matrix, weights):
    n = matrix.shape[0]
    lambda_max = np.dot(matrix.sum(axis=1), weights)
    ci = (lambda_max - n) / (n - 1)
    ri_table = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.9, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45}
    
    ri = ri_table.get(n, 1.45)  # Default value for n > 9
    cr = ci / ri if ri != 0 else 0
    return cr

def ahp_algorithm(criteria, comparisons):
    pairwise_matrix = create_pairwise_comparison_matrix(criteria, comparisons)
    normalized_matrix = normalize_matrix(pairwise_matrix)
    weights = calculate_weights(normalized_matrix)
    
    cr = calculate_consistency_ratio(pairwise_matrix, weights)
    
    if cr < 0.1:
        print("Consistency Ratio (CR):", cr)
        print("Weights of criteria:", weights)
        print("The judgments are consistent.")
    else:
        print("Consistency Ratio (CR):", cr)
        print("The judgments are not consistent. Please revise the pairwise comparisons.")
    
    return weights

# Example criteria and comparisons
criteria = ["Criterion 1", "Criterion 2", "Criterion 3"]
comparisons = {
    (0, 1): 3,  # Criterion 1 is 3 times more important than Criterion 2
    (0, 2): 5,  # Criterion 1 is 5 times more important than Criterion 3
    (1, 2): 2,  # Criterion 2 is 2 times more important than Criterion 3
}

weights = ahp_algorithm(criteria, comparisons)
