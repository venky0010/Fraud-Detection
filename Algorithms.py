import numpy as np
import pandas as pd
from scipy.stats import beta
from scipy.stats import chi2_contingency
from sklearn.metrics import confusion_matrix

def Algorithm1(pmf, PM, PPM, thresh1, thresh2):
    
    contingency_matrix = np.zeros((2, 3))
    
    p = pmf[:, :5]
    contingency_matrix[0, 0] = p[p<thresh1].sum()
    contingency_matrix[0, 1] = p[np.where((p>thresh1) & (p<thresh2))].sum()
    contingency_matrix[0, 2] = p[p>thresh2].sum() + pmf[:, 5].sum()
    
    contingency_matrix[1, 0] = sum([PM[i[0], i[1]] for i in np.argwhere(PPM < thresh1).tolist()])
    contingency_matrix[1, 1] = sum([PM[i[0], i[1]] for i in np.argwhere((PPM > thresh1) & (PPM < thresh2)).tolist()])
    contingency_matrix[1, 2] = sum([PM[i[0], i[1]] for i in np.argwhere(PPM > thresh2).tolist()])
    
    #print(contingency_matrix)
    if sum(contingency_matrix[0])==0 or sum(contingency_matrix[1])==0 or sum(contingency_matrix[:, 0])==0 or sum(contingency_matrix[:, 1])==0 or sum(contingency_matrix[:, 2])==0:
        return 2000, 0
    
    chi2, p, dof, ex = chi2_contingency(contingency_matrix, correction=False)
    
    #print("Algorithm 1",chi2, p, dof, ex) contingency_matrix
    return p , contingency_matrix

def Algorithm2(pmf, PM, PPM):
    
    contingency_matrix = np.zeros((2, 2))
    
    contingency_matrix[0, 0] = pmf[:, :5].sum()
    contingency_matrix[0, 1] = pmf[:, 5].sum()
    
    contingency_matrix[1, 0] = PM[:, :5].sum()
    contingency_matrix[1, 1] = PM[:, 5].sum()
    #print(contingency_matrix)
    if sum(contingency_matrix[0])==0 or sum(contingency_matrix[1])==0 or sum(contingency_matrix[:, 0])==0 or sum(contingency_matrix[:, 1])==0:
        return 2000
    
    chi2, p, dof, ex = chi2_contingency(contingency_matrix, correction=False)
    
    #print("Algorithm 2",chi2, p, dof, ex)
    return p

def Algorithm3(S1, S2, S3, S4, PMP, thresh1, thresh2):   
    contingency_matrix = np.zeros((2, 3))
    
    p = S1[:, :5]
    #print(p)
    #print(p[p<thresh1].sum(), p[np.where((p>thresh1) & (p<thresh2))].sum(), p[p>thresh2].sum() + S1[:, 5].sum())
    
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
    #print(index)
    contingency_matrix[1, 1] = sum([S3[i[0], i[1]] for i in index])
    contingency_matrix[1, 2] = sum([S3[i[0], i[1]] for i in np.argwhere(S4 > thresh2).tolist()])
    #print(sum([S3[i[0], i[1]] for i in np.argwhere(S4 < thresh1).tolist()]), sum([S3[i[0], i[1]] for i in index]), sum([S3[i[0], i[1]] for i in np.argwhere(S4 > thresh2).tolist()]))
    #print(contingency_matrix)
    if sum(contingency_matrix[0])==0 or sum(contingency_matrix[1])==0 or sum(contingency_matrix[:, 0])==0 or sum(contingency_matrix[:, 1])==0 or sum(contingency_matrix[:, 2])==0:
        #print("Yes")
        return 2000
    
    chi2, p, dof, ex = chi2_contingency(contingency_matrix, correction=False)
    #print("Algo 3 p value", p)
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
  
def Staging(ca, wa, PM, PMF):
    
    staging1 = wa*PMF[:, :5]
    staging1 = np.column_stack((staging1, PMF[:, 5]))
    staging2 = ca*PM[:, :5]
    staging2 = np.column_stack((staging2, list(np.logical_not(sum(staging2.T)).astype(int))))

    staging3 = []
    for i in range(len(PM)):
        staging3.append(PM[i, :]*staging2[:, 5][i])
    staging3 = np.array(staging3)

    staging4 = staging3*PMF
    
    return staging1, staging2, staging3, staging4

def Match(R1, R2, pmf):
    
    pairs_match = [[0]*6 for i in range(len(R1))]
    
    for i in range(len(R1)):
        
        if R1[i] == R2[i]:
            
            pairs_match[i][R1[i]] = 1
        
        else:
            
            pairs_match[i][-1] = 1
    
    pair_match = np.array(pairs_match)
    pair_match_prob = pair_match*pmf
    
    return pair_match, pair_match_prob, pmf

#Main function which requires pair's answer sequences and other parameters (pmf, correct answerbinary matrix, wrong answer binary matrix, thresholds)
def Main(R1, R2, pmf, ca, wa, thresh1, thresh2):   #R1, R2 are actual sequence
    
    PM, PMP, PMF = Match(R1, R2, pmf)
    S1, S2, S3, S4 = Staging(ca, wa, PM, PMF)
    
    p1 = Algorithm1(PMF, PM, PMP, thresh1, thresh2)
    p2 = Algorithm2(PMF, PM, PMP)
    
    p3 = Algorithm3(S1, S2, S3, S4, PMP, thresh1, thresh2)
    p4 = Algorithm4(S1, S3)
    
    return p1, p2, p3, p4
  
  
def correctAnswers(data):
       
    CA = {}                                         # Key: Question, Values: correct Answer
    for i in range(len(data)):
        question = data.loc[i, 'q_id']
        if question not in CA:
            CA[question] = data.loc[i, 'choice id']    

    return CA

#To be used for main algorithm and not simulation

def get_sequence(response1, response2, PMF):
    
    r1, r2 = [], []
    pmf = []
    for que, ans in response1.items():
        if que in response2:
            r1.append(ans)
            r2.append(response2[que])
            pmf.append(list(PMF[que].values()))
            
    return r1, r2, np.array(pmf)
  
def studentResponses(data, correctAnswers):
    
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
  
def gettingDistributions(correctAnswers, allResponsesWithStudentId, distribution):
    
    globalProbabilities = {i: {0:0, 1:0, 2:0, 3:0, 4:0} for i in correctAnswers}
    pmf = {i: {0:0, 1:0, 2:0, 3:0, 4:0, 'mismatch':0} for i in correctAnswers}
    cdf = {i: {0:0, 1:0, 2:0, 3:0, 4:0, 'mismatch':0} for i in correctAnswers}
    
    for question, response in distribution.items():
        s = 0
        for i in response:
            globalProbabilities[question][i] = response[i]/sum(response.values())
            s = (response[i]/sum(response.values()))**2
            pmf[question][i] = s
            cdf[question][i] += s
        pmf[question]['mismatch'] = 1 - sum(pmf[question].values())
        cdf[question]['mismatch'] = sum(pmf[question].values())
    
    pmf = np.array([list(answer.values()) for ques, answer in pmf.items()])
    cdf = np.array([list(answer.values()) for ques, answer in cdf.items()])
    
    return pmf, cdf
  
 def Sections(data):
    #Go through every section/syb-unit and save the student's username in a dictionary
    allSections = {}    
    data = df1.copy()
    DISTRICTS = data.groupby(['district_name'])
    for district_name, item in DISTRICTS:

        district = DISTRICTS.get_group(district_name)
        BLOCKS = district.groupby(['block_name'])
    
        for block_name, value in BLOCKS:
        
            block = BLOCKS.get_group(block_name)
            EDU_DISTS = block.groupby(['edu_dist_name'])
        
            for edu_name, value in EDU_DISTS:
            
                edu_dist = EDU_DISTS.get_group(edu_name)
                SCHOOLS = edu_dist.groupby(['school_name'])
            
                for school_name, value in SCHOOLS:
                
                    school = SCHOOLS.get_group(school_name)
                    SECTIONS = school.groupby(['Section'])
                
                    username = []
                    for section_name, value in SECTIONS:
                    
                        #print(district_name, block_name, edu_name, school_name, section_name)
                        section = SECTIONS.get_group(section_name)
                        #print(len(section), len(set(section['emis_username'].tolist())))
                        name = district_name+"_"+block_name+"_"+edu_name+"_"+school_name+"_"+section_name
                        allSections[name] = section['emis_username'].tolist()
                        
    return allSections
