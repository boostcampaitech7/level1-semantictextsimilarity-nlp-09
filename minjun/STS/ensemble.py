import pandas as pd
import numpy as np
import sys

def find_closest_label(predicted_value):
    if predicted_value < 0:
        value = 0.0
    elif predicted_value == -0.0:
        value = 0.0
    elif predicted_value > 5:
        value = 5.0
    elif (predicted_value * 10) % 2 != 0:
        labels = np.array([0., 0.2, 0.4, 0.5, 0.6, 0.8, 1., 1.2, 1.4, 1.5, 1.6, 1.8, 2.,
                           2.2, 2.4, 2.6, 2.8, 3., 3.2, 3.4, 3.5, 3.6, 3.8, 4., 4.2, 4.4,
                           4.5, 4.6, 4.8, 5.])
        counts = np.array([21, 7, 16, 2, 20, 22, 22, 27, 17, 4, 18, 22, 22, 23, 21, 22, 22,
                           22, 22, 22, 3, 19, 22, 22, 25, 19, 7, 15, 22, 22])
        distances = np.abs(labels - predicted_value)
        weighted_distances = distances / counts
        # 가장 가까운 라벨 선택
        value = labels[np.argmin(weighted_distances)]
    else:
        value = predicted_value
    return float(value)



def soft_voting(dfs):
    
    soft_voted_preds = np.mean([df['target'].values for df in dfs], axis=0)
    #희준쓰  
    print(dfs)
    soft_voted_preds = np.round(soft_voted_preds, decimals=1)
    soft_voted_preds = np.array([find_closest_label(pred) for pred in soft_voted_preds])
    
    soft_voted_df = pd.DataFrame({
        'id': dfs[0]['id'],
        'target': soft_voted_preds
    })
    
    return soft_voted_df



def weight_voting(dfs, weights):

    weight_voted_preds = np.average([df['target'].values for df in dfs], axis=0, weights=weights)



    weight_voted_preds = np.round(weight_voted_preds, decimals=1)
    weight_voted_preds = np.array([find_closest_label(pred) for pred in weight_voted_preds])
    weight_voted_df = pd.DataFrame({
        'id': dfs[0]['id'],
        'target': weight_voted_preds
    })
    
    return weight_voted_df



if __name__ == "__main__":
    
    file_paths = ['./0.9330412_snunlp-KR-ELECTRA-discriminator_output.csv',
                  "./DH_KrELRCTRA_9302.csv",
                  "./output_hun.csv",
                  "./output_1.csv",
                  "./output_3.csv"]
    
    
    
    
    dfs = [pd.read_csv(path) for path in file_paths]
    
    weights = [0.2,0.3,0.2]
    
    voting_method = sys.argv[1]
    
    if voting_method == "soft-voting":
        final_output = soft_voting(dfs)
        filename = "soft_voted_final_output.csv"
    elif voting_method == "weight-voting":
        final_output = weight_voting(dfs, weights)
        filename = "weight_voted_final_output"
    else:
        print("Use 'soft-voting' or 'weight-voting'")
        sys.exit(1)
        
        
    final_output.to_csv(f"./{filename}", index=False)


