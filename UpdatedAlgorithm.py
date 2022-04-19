import numpy as np
import pandas as pd

def Algorithm1(pmf, PM, PPM, thresh1, thresh2):
    
    contingency_matrix = np.zeros((2, 3))
    
    p = pmf[:, :5]
    contingency_matrix[0, 0] = p[p<thresh1].sum()
    contingency_matrix[0, 1] = p[np.where((p>thresh1) & (p<thresh2))].sum()
    contingency_matrix[0, 2] = p[p>thresh2].sum() + pmf[:, 5].sum()
    
    contingency_matrix[1, 0] = sum([PM[i[0], i[1]] for i in np.argwhere(PPM < thresh1).tolist()])
    contingency_matrix[1, 1] = sum([PM[i[0], i[1]] for i in np.argwhere((PPM > thresh1) & (PPM < thresh2)).tolist()])
    contingency_matrix[1, 2] = sum([PM[i[0], i[1]] for i in np.argwhere(PPM > thresh2).tolist()])
    
    if sum(contingency_matrix[0])==0 or sum(contingency_matrix[1])==0 or sum(contingency_matrix[:, 0])==0 or sum(contingency_matrix[:, 1])==0 or sum(contingency_matrix[:, 2])==0:
        return 2000
    
    chi2, p, dof, ex = chi2_contingency(contingency_matrix, correction=False)
    
    #print("Algorithm 1",chi2, p, dof, ex)
    return p 

def Algorithm2(pmf, PM, PPM):
    
    contingency_matrix = np.zeros((2, 2))
    
    contingency_matrix[0, 0] = pmf[:, :5].sum()
    contingency_matrix[0, 1] = pmf[:, 5].sum()
    
    contingency_matrix[1, 0] = PM[:, :5].sum()
    contingency_matrix[1, 1] = PM[:, 5].sum()
    
    if sum(contingency_matrix[0])==0 or sum(contingency_matrix[1])==0 or sum(contingency_matrix[:, 0])==0 or sum(contingency_matrix[:, 1])==0:
        return 2000
    
    chi2, p, dof, ex = chi2_contingency(contingency_matrix, correction=False)
    
    #print("Algorithm 2",chi2, p, dof, ex)
    return p

def Algorithm3(S1, S2, S3, S4, PMP, thresh1, thresh2):   
    contingency_matrix = np.zeros((2, 3))
    
    p = S1[:, :5]
    contingency_matrix[0, 0] = p[p<thresh1].sum()
    contingency_matrix[0, 1] = p[np.where((p>thresh1) & (p<thresh2))].sum()
    contingency_matrix[0, 2] = p[p>thresh2].sum() + S1[:, 5].sum()
    contingency_matrix[1, 0] = sum([S3[i[0], i[1]] for i in np.argwhere(S4 < thresh1).tolist()])
    
    indexes1 = np.argwhere((S4 > thresh1)).tolist()
    indexes2 = np.argwhere((PMP < thresh2)).tolist()
    index = []
    for i in indexes1:
        if i in indexes2:
            index.append(i)
    
    contingency_matrix[1, 1] = sum([S3[i[0], i[1]] for i in index])
    contingency_matrix[1, 2] = sum([S3[i[0], i[1]] for i in np.argwhere(S4 > thresh2).tolist()])
    
    #print(contingency_matrix)
    if sum(contingency_matrix[0])==0 or sum(contingency_matrix[1])==0 or sum(contingency_matrix[:, 0])==0 or sum(contingency_matrix[:, 1])==0 or sum(contingency_matrix[:, 2])==0:
        return 2000
    
    chi2, p, dof, ex = chi2_contingency(contingency_matrix, correction=False)
    
    #print("Algorithm 3",chi2, p, dof, ex)
    return p

def Algorithm4(S1, S3):
    
    contingency_matrix = np.zeros((2, 2))
    
    contingency_matrix[0, 0] = S1[:, :5].sum()
    contingency_matrix[0, 1] = S1[:, 5].sum()
    
    contingency_matrix[1, 0] = S3[:, :5].sum()
    contingency_matrix[1, 1] = S3[:, 5].sum()
    
    #print(contingency_matrix)
    
    if sum(contingency_matrix[0])==0 or sum(contingency_matrix[1])==0 or sum(contingency_matrix[:, 0])==0 or sum(contingency_matrix[:, 1])==0:
        return 2000
    
    chi2, p, dof, ex = chi2_contingency(contingency_matrix, correction=False)
    
    #print("Algorithm 4",chi2, p, dof, ex)
    return p
