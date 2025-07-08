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
import numpy as np
import pandas as pd 
from itertools import product  


###################################### Noise imputation - approach 2 ###################################### 

def state_to_index(state, m):
    """Optimized index calculation"""
    return sum(s * (5 ** (m - 1 - i)) for i, s in enumerate(state))


def index_to_state(index, m):
    """More efficient state reconstruction"""
    return tuple((index // (5 ** i)) % 5 for i in reversed(range(m)))


def apply_stage_noise(state, epsilon):
    """Generate all possible distorted states with probabilities"""
    n_stages = len(state)
    mislabel_prob = epsilon / 4
    
    for distortion_mask in product([0, 1], repeat=n_stages):
        distorted_state = []
        prob = 1.0
        for i, (s, mask) in enumerate(zip(state, distortion_mask)):
            if mask == 0:  # Correct label
                distorted_state.append(s)
                prob *= (1 - epsilon)
            else:  # Mislabeled
                # Cycle through alternatives deterministically
                alt = (s + 1 + (i % 3)) % 5  # Ensures different stage
                distorted_state.append(alt)
                prob *= mislabel_prob
        yield tuple(distorted_state), prob


def distort_stp_matrix(P_m, epsilon, m):
    """Optimized matrix distortion"""
    n_states, n_next = P_m.shape
    P_noisy = np.zeros_like(P_m)
    
    for from_idx in range(n_states):
        from_state = index_to_state(from_idx, m)
        
        # Apply noise to history stages
        for distorted_state, hist_prob in apply_stage_noise(from_state, epsilon):
            distorted_idx = state_to_index(distorted_state, m)
            
            # Apply noise to next stage
            for next_stage in range(n_next):
                for distorted_next, next_prob in apply_stage_noise((next_stage,), epsilon):
                    P_noisy[distorted_idx, distorted_next[0]] += (
                        P_m[from_idx, next_stage] * hist_prob * next_prob
                    )
    
    # Normalize with safety checks
    row_sums = P_noisy.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Prevent division by zero
    return P_noisy / row_sums


def impute_noise(mdf, epsilon, stage_type):
    """Optimized DataFrame processing"""
    shape_map = {2: (5,5), 3: (25,5), 4: (125,5)}
    n_rows, n_cols = shape_map[stage_type]
    # df = mdf.iloc[:, 3:] 
    df = mdf
    
    if df.shape[1] != n_rows * n_cols:
        raise ValueError(f"Expected {n_rows*n_cols} columns for stage_type={stage_type}")
    
    # Vectorized reshaping and processing
    def process_row(row):
        # mat = row.values.reshape(n_rows, n_cols)
        mat = row.reshape(n_rows, n_cols)
        noisy = distort_stp_matrix(mat, epsilon, stage_type - 1)
        return noisy.ravel()
    
    tdf = pd.DataFrame(
        np.apply_along_axis(process_row, 1, df),
        index=df.index,
        columns=df.columns
    )
    # tdf = pd.concat([mdf.iloc[:, :3], tdf], axis=1) 
    return tdf 


def impute_noise_for_dataset(df, epsilon):
    """Optimized DataFrame processing"""
    # shape_map = {2: (5,5), 3: (25,5), 4: (125,5)}
    # n_rows, n_cols = shape_map[stage_type]
    col = df.columns.tolist() 
    col_dict = {k:[] for k in range(4)} 
    for c in col:
        col_dict[c.count("->")].append( c )  
    
    noisy_df = df[col_dict[0]] 
    for i in range(1, 4): 
        sel_col = col_dict[i] 
        stage_type = i+1 
        tdf = impute_noise(df[sel_col], epsilon, stage_type) 
        noisy_df = pd.concat([noisy_df, tdf], axis=1) 
    return noisy_df 





