import numpy as np
import pandas as pd
from scipy.stats import beta
import matplotlib.pyplot as plt


#Below code is run on original data to fit beta and normal distribution for the simulation part
def correctChoices(data):
    
    correctAnswers = {}
    for i in range(len(data)):
        question = data.loc[i, 'q_id']
        if question not in correctAnswers:
            correctAnswers[question] = data.loc[i, 'choice id']    

    return correctAnswers
  
  
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
    
    return allResponsesWithStudentId
  
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
  
  
#Below code if for simulation

def proportion(students, CA, responses):
    
    pro = []
    for student in students:
        
        c = 0
        for ques, ans in responses[student].items():
            
            if ans == CA[ques]:
                c+=1
        
        pro.append(c)
        
    return pro
  
beta_class = {}   #Beta Dist Parameteres
section_proportion = {} #Proportion of correct Answers in each section
sections_beta = {}      #Saves parameters for class

correctAnswers = correctChoices(df2)                  #Correct Answers to all questions
allStudents = studentResponses(df1, correctAnswers)   #Response of all students in file using emis_username as key identifier    
allSections = Sections(df1)                           #Save each section with student usernames in it

for section, students in allSections.items():
    
    value = proportion(students, correctAnswers, allStudents)
    section_proportion[section] = value
    #print(len(students), len(value))
    sections_beta[section] = list(beta.fit(value))[0:2]
    
    
def get_response(answer, CA):
    
    for i in range(len(answer)):
        if answer[i] == 1:
            answer[i] = CA[i+1]
        else:
            choose = [0, 1, 2, 3, 4]
            choose.remove(CA[i+1])
            answer[i] = np.random.choice(choose)
            
    answer = {i+1 : answer[i] for i in range(len(answer))}
    return answer
  
CA = {i: np.random.randint(0, 5) for i in range(1, 51)}
studentResponses = {}

def Simulation(section, CA):
    
    section_sim = {}
        
    for i in section:
         
        alpha, beta = list(sections_beta.values())[np.random.choice(len(sections_beta))]
        students = np.random.beta(alpha, beta, size=len(i))
        section_sim[i] = students
        
        for student in students:
            
            proportion = student
            student_response = [np.random.random() for j in range(50)]
            ans = [1 if k<=proportion else 0 for k in student_response]
            ans = get_response(ans, CA)
            studentResponses[student] = ans
            
    return section_name, studentResponses
  
section_sim, studentResponseswithID = Simulation(list(section_proportion.keys()), CA)
