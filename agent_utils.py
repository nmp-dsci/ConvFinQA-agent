
from pydantic import BaseModel, Field, validator, ValidationError
import pandas as pd 
import numpy as np 
import anthropic 
import os,json, re 
from dotenv import load_dotenv


def training_data():
    with open( "data/convfinqa_dataset.json") as f:
        data = json.load(f)
    
    ## structure of the data 
    for key in data.keys():  
        globals()[f'{key}_df'] = pd.concat([pd.DataFrame({**x['dialogue'], 'data_key': key, 'report_id': x['id'],'q_order':range(len(x['dialogue']['conv_questions']))}) for x in data[key ]], ignore_index=True)
        globals()[f'{key}_features'] = pd.DataFrame([{**x['features'],**{'report_id': x['id'],'data_key': key}} for x in data[key ]]) 
   
    ## get report question features 
    features_df = pd.concat([globals()[f'{key}_features'] for key in data.keys() ],ignore_index=True)
    features_df['has_type2_question'].apply(lambda x: 'Simple' if x == False else 'Complex')
    assert features_df['report_id'].nunique() == features_df.shape[0], "id should be unique for each report"
    ## get report questions 
    question_df = pd.concat([globals()[f'{key}_df'] for key in data.keys() ],ignore_index=True) 
    question_df['base'] = 'Overall'
    question_df['turn_type'] = pd.to_numeric(question_df['turn_program'],errors='coerce').apply(lambda x: 'Number' if pd.notnull(x) else 'Program')
    ## decompossing attributes of `turn_program` to understand the complexity/scope  of the calculations
    question_df['turn_program_actions'] = question_df['turn_program'].str.split('(?<=\)),')
    question_df['turn_program_actions_n']  = question_df['turn_program_actions'].apply(len)
    question_df['turn_program_calcs'] = question_df['turn_program_actions'].apply(lambda x: [ m.group(1) if (m := re.match(r'\s*(\w+)\(', s)) else None for s in x ])
    question_df['question_id'] = question_df['report_id'] + '_q' + question_df['q_order'].astype(str)
    ## merge report answer attirbutes to Report /Question level data
    question_df = question_df.merge(features_df, on=['report_id','data_key'], how='left')
    
    ## check data quality
    assert question_df['question_id'].value_counts().max() == 1 , "question_id should be unique for each question"
    assert question_df.isnull().sum(axis=0).sum() == 0, "There should be no missing values after merge"
    
    ## Test accuracy of Agent AT THIS LEVEL 
    return question_df 


def load_agent_answers(agent_name=''):
    if os.path.exists(f"evaluations/{agent_name}_scoring.jsonl"):
        with open(f"evaluations/{agent_name}_scoring.jsonl", "r") as f:
            agent_answers = [json.loads(line) for line in f]
    else:
        agent_answers = []
        with open(f"evaluations/{agent_name}_scoring.jsonl", "w") as f:
            for entry in agent_answers:
                f.write(json.dumps(entry) + "\n")
    return agent_answers

