#import torch
import numpy as np
import pandas as pd
#torch.cuda.is_available()

#Reading files and storing different distributions

def correctChoices(data):
    
    correctAnswers = {}
    for i in range(len(data)):
        question = data.loc[i, 'q_id']
        if question not in correctAnswers:
            correctAnswers[question] = data.loc[i, 'choice id']    

    return correctAnswers


def studentResponsesAndDistributions(data, correctAnswers):
    
    distribution = {i: {0:0, 1:0, 2:0, 3:0, 4:0} for i in correctAnswers}
    allResponsesWithStudentId = {}
    
    for index in range(len(df1)):
        sID = df1.loc[index, 'emis_username']
        #print(sID)
    
        if sID not in allResponsesWithStudentId:
            allResponsesWithStudentId[sID] = {}
            string = df1.loc[index, 'AnswerString']

            for response in string.split(","):
                question = int(response[:-1])
                answer = int(response[-1])
                allResponsesWithStudentId[sID][question] = answer
                distribution[question][answer]+=1
    
    return allResponsesWithStudentId, distribution


def probabilityDistributions(correctAnswers, allResponsesWithStudentId, distribution):
    
    probabilityDistribution = {i: {0:0, 1:0, 2:0, 3:0, 4:0} for i in correctAnswers}
    pmf = {i: {0:0, 1:0, 2:0, 3:0, 4:0, 'mismatch':0} for i in correctAnswers}
    cdf = {i: {0:0, 1:0, 2:0, 3:0, 4:0, 'mismatch':0} for i in correctAnswers}
    
    for question, response in distribution.items():
        s = 0
        for i in response:
            probabilityDistribution[question][i] = response[i]/sum(response.values())
            s = (response[i]/sum(response.values()))**2
            pmf[question][i] = s
            cdf[question][i] += s
        pmf[question]['mismatch'] = 1 - sum(pmf[question].values())
        cdf[question]['mismatch'] = sum(pmf[question].values())
        
    return probabilityDistribution, pmf, cdf
    


def fileReadingandProcessing(data1, data2):
    
    correctAnswers = correctChoices(data2)
    allResponsesWithStudentId, distribution = studentResponsesAndDistributions(data1, correctAnswers)
    probabilityDistribution, pmf, cdf = probabilityDistributions(correctAnswers, allResponsesWithStudentId, distribution)
    
    return correctAnswers, allResponsesWithStudentId, probabilityDistribution, pmf, cdf
  
xls = pd.ExcelFile('Basic quiz week 4.xlsx')
df1 = pd.read_excel(xls, 'Basic Quiz Week4 Answer Strings')
df2 = pd.read_excel(xls, 'Answers')
correctAnswers, allResponsesWithStudentId, probabilityDistribution, pmf, cdf = fileReadingandProcessing(df1, df2)



#Different Algorithms

from scipy.stats import chi2_contingency
def Algorithm1(pmf, PM, PPM):
    
    contingency_matrix = np.zeros((2, 3))
    
    p = pmf[:, :4]
    contingency_matrix[0, 0] = p[p<0.2].sum()
    contingency_matrix[0, 1] = p[np.where((p>0.2) & (p<0.4))].sum()
    contingency_matrix[0, 2] = p[p>0.4].sum() + pmf[:, 4].sum()
    
    contingency_matrix[1, 0] = sum([PM[i[0], i[1]] for i in np.argwhere(PPM < 0.2).tolist()])
    contingency_matrix[1, 1] = sum([PM[i[0], i[1]] for i in np.argwhere((PPM > 0.2) & (PPM < 0.4)).tolist()])
    contingency_matrix[1, 2] = sum([PM[i[0], i[1]] for i in np.argwhere(PPM > 0.4).tolist()])
    
    chi2, p, dof, ex = chi2_contingency(contingency_matrix, correction=False)
    
    #print("Algorithm 1",chi2, p, dof, ex)
    return p
    
def Algorithm2(pmf, PM, PPM):
    
    contingency_matrix = np.zeros((2, 2))
    print(pmf[pmf<0.2])
    
    contingency_matrix[0, 0] = pmf[:, :4].sum()
    contingency_matrix[0, 1] = pmf[:, 4].sum()
    
    contingency_matrix[1, 0] = PM[:, :4].sum()
    contingency_matrix[1, 1] = PM[:, 4].sum()
    
    chi2, p, dof, ex = chi2_contingency(contingency_matrix, correction=False)
    
    #print("Algorithm 2",chi2, p, dof, ex)
    return p
    
def Algorithm3(IPMF, PPM, S1, S2):
    
    contingency_matrix = np.zeros((2, 3))
    
    p = IPMF[:, :4]
    contingency_matrix[0, 0] = p[p<0.2].sum()
    contingency_matrix[0, 1] = p[np.where((p>0.2) & (p<0.4))].sum()
    contingency_matrix[0, 2] = p[p>0.4].sum() + IPMF[:, 4].sum()
    
    contingency_matrix[1, 0] = sum([S1[i[0], i[1]] for i in np.argwhere(S2 < 0.2).tolist()])
    contingency_matrix[1, 1] = sum([S1[i[0], i[1]] for i in np.argwhere((S2 > 0.2)).tolist()]+[S1[i[0], i[1]] for i in np.argwhere((PPM < 0.4)).tolist()])
    contingency_matrix[1, 2] = sum([S1[i[0], i[1]] for i in np.argwhere(S2 > 0.4).tolist()])
    
    chi2, p, dof, ex = chi2_contingency(contingency_matrix, correction=False)
    
    #print("Algorithm 3",chi2, p, dof, ex)
    return p

def Algorithm4(IPMF, S1):
    
    contingency_matrix = np.zeros((2, 2))
    
    contingency_matrix[0, 0] = IPMF[:, :4].sum()
    contingency_matrix[0, 1] = IPMF[:, 4].sum()
    
    contingency_matrix[1, 0] = S1[:, :4].sum()
    contingency_matrix[1, 1] = S1[:, 4].sum()
    
    chi2, p, dof, ex = chi2_contingency(contingency_matrix, correction=False)
    
    #print("Algorithm 4",chi2, p, dof, ex)
    return p
