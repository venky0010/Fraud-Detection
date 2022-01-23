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
