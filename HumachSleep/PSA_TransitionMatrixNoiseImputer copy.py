'''
######################################
Importing necessary modules
'''
import PSA_Imports 
import os, sys 
import glob, re, copy 
import numpy as np
import pandas as pd 
from itertools import product 
# import matplotlib 
# import matplotlib.pyplot as plt 




###################################### Noise imputation - approach 2 ###################################### 


def state_to_index(state, m):
    """Convert state tuple to flattened index (e.g., (0,1,2) → 7 for m=3)."""
    return sum([state[i] * (5 ** (m - 1 - i)) for i in range(m)])

def index_to_state(index, m):
    """Convert flattened index to state tuple (e.g., 7 → (0,1,2) for m=3)."""
    state = []
    for _ in range(m):
        state.append(index % 5)
        index = index // 5
    return tuple(reversed(state))

def generate_distorted_states(state, percent_noise_per_stage, mislabel_prob):
    """Generate all possible distorted versions of a state tuple with probabilities."""
    possible_states = []
    # Generate all combinations of correct/mislabeled stages
    for distortions in product([0, 1], repeat=len(state)):  # 0=correct, 1=mislabeled
        distorted_state = []
        prob = 1.0
        for i in range(len(state)):
            if distortions[i] == 0:
                distorted_state.append(state[i])
                prob *= (1 - percent_noise_per_stage)
            else:
                # Choose any of the other 4 stages uniformly
                distorted_state.append(np.random.choice([x for x in range(5) if x != state[i]]))
                prob *= mislabel_prob
        possible_states.append((tuple(distorted_state), prob))
    return possible_states


def distort_stp_matrix_per_stage(P_m, percent_noise_per_stage, m):
    """
    Add mislabeling noise to an m-stage STP matrix (shape 5^m × 5).
    
    Args:
        P_m: Input STP matrix (5^m × 5).
        percent_noise_per_stage: Mislabeling rate per stage (e.g., 0.01 for 1% per stage).
        m: Number of stages in the input sequence (e.g., m=2 for 3-stage STP).
    
    Returns:
        P_m_noisy: Noisy STP matrix (same shape as P_m).
    """
    n_states = 5 ** m
    P_m_noisy = np.zeros_like(P_m)
    mislabel_prob = percent_noise_per_stage / 4  # Uniform mislabeling to 4 other stages
    
    for from_idx in range(n_states):
        from_state = index_to_state(from_idx, m)
        
        # Generate all possible observed (noisy) input sequences with probabilities
        distorted_states = generate_distorted_states(from_state, percent_noise_per_stage, mislabel_prob)
        for distorted_state, prob in distorted_states:
            distorted_idx = state_to_index(distorted_state, m)
            P_m_noisy[distorted_idx, :] += P_m[from_idx, :] * prob
    
    # Normalize rows to sum to 1 (handle division by zero)
    row_sums = P_m_noisy.sum(axis=1)
    row_sums[row_sums == 0] = 1  # Avoid division by zero for unused states
    P_m_noisy = P_m_noisy / row_sums[:, np.newaxis]
    return P_m_noisy


'''
###################################### Noise imputation - approach 2 ... 
Reshape STP columns to matrix for noise imputation 
'''
def reshape_stp_columns_to_matrix(df_row, stage_type):
    """
    Reshape a single row's STP values into the appropriate matrix format.
    
    Args:
        df_row: Single row from DataFrame containing all STP values
        stage_type: 2, 3, or 4
    
    Returns:
        STP matrix in correct shape (5×5, 25×5, or 125×5)
    """
    shapes = {
        2: (5, 5),
        3: (25, 5),
        4: (125, 5)
    }
    
    if stage_type not in shapes:
        raise ValueError(f"stage_type must be one of {list(shapes.keys())}")
    
    n_rows, n_cols = shapes[stage_type]
    stp_values = df_row.values.reshape(n_rows, n_cols)
    return stp_values



'''
###################################### Noise imputation - approach 2 ... 
Impute noise in transition matrix after matrix calculation is done for each subject 
'''
def noise_imputation_per_subject(subject_row, percent_noise_per_stage, stage_type):
    """
    Apply distortion to a single subject's STP values.
    
    Args:
        subject_row: Pandas Series with all STP values for one subject
        percent_noise_per_stage: Mislabeling rate per stage
        stage_type: Type of STP
    
    Returns:
        Distorted STP values as a flat array
    """
    # Reshape to proper matrix format
    stp_matrix = reshape_stp_columns_to_matrix(subject_row, stage_type)
    
    # Apply distortion
    m = {2: 1, 3: 2, 4: 3}[stage_type]
    distorted_matrix = distort_stp_matrix_per_stage(stp_matrix, percent_noise_per_stage, m)
    
    # Flatten back to original shape
    return distorted_matrix.reshape(-1)


'''
###################################### Noise imputation - approach 2 ... 
Impute noise in transition matrix after matrix calculation is done for a dataframe 
'''
def impute_noise_in_transitiom_matrix(all_annot_df, percent_noise_per_stage=0.01, stage_type=2):
    """
    Apply mislabeling noise to all subjects' STP values in a DataFrame.
    
    Args:
        all_annot_df: DataFrame where each row is a subject and columns are STP values
        percent_noise_per_stage: Mislabeling rate per stage (e.g., 0.01 for 1%)
        stage_type: Type of STP (2, 3, or 4)
    
    Returns:
        DataFrame with distorted STP values (same shape as input)
    """
    # Validate stage type
    if stage_type not in [2, 3, 4]:
        raise ValueError("stage_type must be 2, 3, or 4")
    
    # Calculate expected columns
    expected_cols = {
        2: 5 * 5,
        3: 25 * 5,
        4: 125 * 5
    }[stage_type]
    
    if all_annot_df.shape[1] != expected_cols:
        raise ValueError(f"For {stage_type}, expected {expected_cols} columns, got {all_annot_df.shape[1]}")
    
    # Apply distortion to each subject
    distorted_data = all_annot_df.apply(
        lambda row: noise_imputation_per_subject(row, percent_noise_per_stage, stage_type),
        axis=1,
        result_type='expand'
    )
    
    # Preserve original column names
    distorted_df = pd.DataFrame(
        distorted_data.values,
        index=all_annot_df.index,
        columns=all_annot_df.columns
    )
    
    return distorted_df



