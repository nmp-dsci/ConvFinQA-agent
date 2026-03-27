'''

Test LLM agent v2: plan / multi-agent 
## Agent1: Plan & Answer   
## Reflection: on answer and improve  
## Tools: to control output 
## Chain of though / golden expands 


'''

from pydantic import BaseModel, Field, validator, ValidationError
import pandas as pd 
import numpy as np 
import anthropic 
import os,json, re ,sys
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from agent_utils import training_data,load_agent_answers
from agent_v2 import * 


pd.options.display.max_columns  = 100

load_dotenv()
agent_name = 'agent_v2'

##########################
## Pull data , quick exploration 

question_df = training_data()
question_df.groupby(['has_type2_question','qa_split','turn_type'])['question_id'].size().reset_index(name='count')

######################

'''
---- TEST AGENT --- 
record_agent = recordQAAgent()
record_agent._init_record('Double_GIS/2008/page_83.pdf') 
result1 = record_agent.query_agent1('what was the total of net assets in 2008?') 
result2 = record_agent.query_agent2() 
'''




def agent_v2_backtest(sample_n=10 , seed=42 ):
    """
    Backtest the agent  for Evaluation of performance. 
    """
    ##Define data set to score 
    with open( "data/convfinqa_dataset.json") as f:
        data = json.load(f)
    question_df = training_data()
    ## pull records to score
    np.random.seed(seed)
    reports_score = np.random.choice(question_df.query('data_key=="train"').report_id.unique(), size=sample_n, replace=False)
    train_set = question_df.query(f'report_id in {list(reports_score)}')
    train_set.groupby(['has_type2_question','turn_type'])['question_id'].size().reset_index(name='count')
    ## Init Agent
    record_agent = recordQAAgent()
    # report_id = 'Double_GIS/2008/page_83.pdf'
    for report_id in reports_score : 
        print('-'*50)
        print('Processing report_id:', report_id)
        record_agent._init_record(report_id) 
        ## pull data
        query_data = [x for x in data.get('train') if x['id'] == report_id]
        query_data = query_data[0]
        questions = question_df.query(f'report_id == "{report_id}"')
        # q = questions.conv_questions.to_list()[0]
        for q in questions.conv_questions.to_list():
            #### Agent 1: Planner and Execute
            agent_step = f'{agent_name}_i1'
            qid = f'{record_agent.report_id}_q{record_agent.question_n}'
            agent_answers = load_agent_answers(agent_name=f'{agent_step}')
            if any(a['question_id'] == qid for a in agent_answers):
                print(f"ITER1:Report & Question already answered")
                continue
            result = record_agent.query_agent1(q) 
            with open(f"evaluations/{agent_step}_scoring.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")
            #### Agent 2: Review and Validation
            agent_step = f'{agent_name}_i2'
            agent_answers = load_agent_answers(agent_name=f'{agent_step}')
            if any(a['question_id'] == qid for a in agent_answers):
                print(f"ITER2:Report & Question already answered")
                continue     
            result = record_agent.query_agent2() 
            with open(f"evaluations/{agent_step}_scoring.jsonl", "a") as f:
                f.write(json.dumps(result) + "\n")       



#######################
## EVALUATION

def agent_v2_evalution(agent_id='agent_v2_i1'):
    ## Load agent answers
    with open(f"evaluations/{agent_id}_scoring.jsonl", "r") as f:
        answer_df = pd.DataFrame([json.loads(line) for line in f]) 
    
    ### Build evaluation set and score
    pred_cols = ['executed_answers','turn_program','turn_type','conv_answers','qa_split']
    answer_df =answer_df.rename(columns=dict(zip( pred_cols, [f'{col}_{agent_id}' for col in pred_cols]  )))
    
    ## Create master dataset with agent answers 
    eval_df = question_df.copy() 
    eval_df = eval_df.merge(answer_df,on=['report_id','question_id','conv_questions'], how='left'  )
    eval_df['agent_answered'] = eval_df[f'conv_answers_{agent_id}'].notnull().apply(lambda x: 'Answered' if x else 'Not Answered')
    
    ## Accurary: Predicted vs actual
    eval_df[f'executed_answers_{agent_id}_score'] = (
        pd.to_numeric(eval_df['executed_answers'], errors='coerce').round(3) ==
        pd.to_numeric(eval_df[f'executed_answers_{agent_id}'], errors='coerce').round(3)).astype(int)
    for col in np.setdiff1d(pred_cols, ['executed_answers']):
        eval_df[f'{col}_{agent_id}_score'] = (eval_df[col] == eval_df[f'{col}_{agent_id}']).astype(int)
    
    ## Evaluation: summary 
    eval_metrics = {'question_id':np.size,**dict(zip( [f'{col}_{agent_id}_score' for col in pred_cols if col in ['executed_answers','turn_program','turn_type','qa_split'] ], ['mean']*4 ))}
    eval_summ = eval_df.groupby(['agent_answered']).agg(eval_metrics).reset_index().rename(columns={'agent_answered': 'Methods'})
    for attr in ['turn_type','has_type2_question','q_order']:
        attr_summ = eval_df.groupby(['agent_answered',attr]).agg(eval_metrics).loc['Answered'].reset_index().rename(columns={attr: 'Methods'})
        attr_summ['Methods'] = attr + '-'+attr_summ['Methods'].astype(str) 
        eval_summ = pd.concat([eval_summ, attr_summ], ignore_index=True)   
    eval_summ['agent_id'] = agent_id
    
    ## Performance Vs Paper: comparison with paper performance TABLE 4/5 from the paper to compare against agent 
    with open(f"evaluations/paper_performance.jsonl", "r") as f:
        performace_bar = pd.DataFrame([json.loads(line) for line in f]) 
    
    final_summ = performace_bar.merge(eval_summ, on=['Methods'], how='left').fillna(0)
    final_summ['agent_id'] = agent_id
    # order_cols = sorted([f'{x}_score' for x in pred_cols] + [f'{x}_{agent_id}_score' for x in pred_cols])
    # final_summ[['Methods'] + order_cols]
    
    return eval_df, final_summ


## 

if __name__ == "__main__":
    agent_v2_backtest(sample_n=190, seed=42)
    eval_df_v2_i1, final_summ_v2_i1 = agent_v2_evalution(agent_id='agent_v2_i1')
    eval_df_v2_i2, final_summ_v2_i2 = agent_v2_evalution(agent_id='agent_v2_i2')
    pd.concat([final_summ_v2_i1, final_summ_v2_i2], ignore_index=True).to_csv(f'evaluations/{agent_name}_final_summary.csv', index=False)




# ### Agent performance  

# eval_summ = eval_df.groupby(['agent_answered']).agg(eval_metrics).reset_index().rename(columns={'agent_answered': 'Methods'})


# eval_df.query(f'agent_answered=="Answered" and turn_type_{agent_name}_score == 0')[['conv_questions','turn_type',f'turn_type_{agent_name}_score']]

# eval_df.query(f'agent_answered=="Answered" and executed_answers_{agent_name}_score == 0')[['conv_questions','turn_program',f'turn_program_{agent_name}']]


# eval_df.query(f'agent_answered=="Answered"').groupby(['turn_type','has_type2_question']).agg(eval_metrics)

# eval_df.query(f'agent_answered=="Answered" and has_type2_question == True and executed_answers_{agent_name}_score == 0')


# for i in eval_df.query('report_id == "Double_GIS/2008/page_83.pdf" ' )['conv_questions']: 
#     print(i)





# ## comparison with paper performance TABLE 4/5 from the paper to compare against agent 
# with open(f"evaluations/paper_performance.jsonl", "r") as f:
#     performace_bar = pd.DataFrame([json.loads(line) for line in f]) 

# final_summ = performace_bar.merge(eval_summ, on=['Methods'], how='left').fillna(0)

# final_summ.plot(x='Methods', y=['executed_answers_score',f'executed_answers_{agent_name}_score'], kind='bar', title='Executed Answers Score Comparison', legend=True)
# plt.show() 

# order_cols = sorted([f'{x}_score' for x in pred_cols] + [f'{x}_{agent_name}_score' for x in pred_cols])
# final_summ[['Methods'] + order_cols]



# ###############
# ## Find test cases to improve agent for v2 

# eval_df.query(f'agent_answered == "Answered" and turn_type == "Program" and executed_answers_agent_v1_score == 0 '
# )[['question_id','conv_questions_x','executed_answers', f'executed_answers_{agent_name}','turn_program', f'turn_program_{agent_name}']]


# eval_df.query(f'agent_answered == "Answered" and turn_type == "Program" and turn_program_agent_v1_score == 0 '
# )[['question_id','conv_questions_x','turn_program', f'turn_program_{agent_name}']]

# eval_df.query(f'agent_answered == "Answered"').to_csv('evaluations/agent_v1_detailed_results.csv', index=False)


