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

#Staging and computations to before calling algorithm functions
def Computation(section, pmf, cdf, responses, answers):
    
    for i in section.index[:-1]:
        for j in section.index[1:]:
            
            print(i, j)
            student1 = section.loc[i, 'emis_username']
            student2 = section.loc[j, 'emis_username']
            print(student1, student2)
            student1_sequence = responses[student1]
            student2_sequence = responses[student2]
            #print(student1_sequence)
            #print(student2_sequence)
            
            #result_algo1 = Staging(student1_sequence, student2_sequence, pmf, cdf, answers)
            r1, r2, r3, r4 = Staging(student1_sequence, student2_sequence, answers, pmf, cdf)
            result.append((student1, student2, r1, r2, r3, r4))
            
    return result 
        
        
def Staging(response1, response2, correctAnswers, pmf, cdf):
    
    CPCA = []                  #Current paper's correct answers, saves True boolean value for right option
    PMF  = []                  #Current paper's pmf
    CPCA_inverse = []          #saves True boolean values for wrong options and False for correct option
    PM   = []                  #Pair's match on answers, saves True boolean value for matching option **
    PPM  = []                  #Pair's matching PMF **
    inversePMF = []            #PMF for wrong answers
    
    bits = []
    
    for ques, ans1 in response1.items():
        
        if ques in response2:                          # if question match -> solve, else: skip
            
            x = [0]*5
            x[correctAnswers[ques]-1] = 1              #if correct option in 2, then appends list [0, 1, 0, 0, 0]
            CPCA.append(x)
            CPCA_inverse.append([1-i for i in x])
            PMF.append(list(pmf[ques].values())[1:])
            
            if ans1 == response2[ques]:                #check if both have same answers
                
                y = [0]*5
                y[ans1-1] = 1
                PM.append(y)                           # this y might not be same as x, depends if selected option is same as correct option
                
                if ans1 == correctAnswers[ques]:
                    
                    bits.append(0)
                
                else:
                    
                    bits.append(1)
                
            else:
                
                PM.append([0, 0, 0, 0, 1])
                bits.append(1)
                
        
                
    PPM = [[PMF[j][i]*PM[j][i] for i in range(len(PM[j]))] for j in range(len(PMF))]
    inversePMF = [[PMF[j][i]*(1-CPCA[j][i]) for i in range(len(CPCA[j]))] for j in range(len(PMF))]
    
    #print(len(PM), len(CPCA), len(CPCA_inverse), len(PPM), len(inversePMF), len(bits))
    #print("\nPairs Match", PM[:5])
    #print("\ncorrectAnswers", CPCA[:5])
    #print("Bits", bits)
    #print("\nWrong Answers", CPCA_inverse[:5])
    #print("\nmatching probability", PPM[:5])
    #print("\nwrong answer probability", inversePMF[:5])
    
    staging1 = [[bits[i]*PM[i][j] for j in range(len(PM[i]))] for i in range(len(PM))]                 #Stores bits for ignored answers
    staging2 = [[PMF[j][i]*staging1[j][i] for i in range(len(staging1[j]))] for j in range(len(PMF))]  #PMF values for ignored answers
    
    #print("\nstaging1", staging1[:10])
    #print("\nstaging2", staging2[:10])
    algo1 = Algorithm1(np.array(PMF), np.array(PM), np.array(PPM))
    algo2 = Algorithm2(np.array(PMF), np.array(PM), np.array(PPM))
    algo3 = Algorithm3(np.array(inversePMF), np.array(PPM), np.array(staging1), np.array(staging2))
    algo4 = Algorithm4(np.array(inversePMF), np.array(staging1))
    
    return algo1, algo2, algo3, algo4
    
#Final Block which runs staging and Algorithms by grouping sections across all schools and regions
RESULTS = {}
#Calculate Probability Distribution
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
                
                for section_name, value in SECTIONS:
                    
                    print(district_name, block_name, edu_name, school_name, section_name)
                    section = SECTIONS.get_group(section_name)
                    print(len(section), len(set(section['emis_username'].tolist())))
                    name = district_name+"_"+block_name+"_"+edu_name+"_"+school_name+"_"+section_name
                    
                    if name not in section_results:
                        
                        result = Computation(section, pmf, cdf, allResponsesWithStudentId, correctAnswers)
                        RESULTS[name] = result

#This is the end of the code
