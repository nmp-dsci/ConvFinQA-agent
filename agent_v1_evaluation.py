'''

Test LLM agent v1: single agent 


'''



from pydantic import BaseModel, Field, validator, ValidationError
import pandas as pd 
import numpy as np 
import anthropic 
import os,json, re 
from dotenv import load_dotenv
import matplotlib.pyplot as plt

pd.options.display.max_columns  = 100

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


agent_name = 'agent_v1'
model_name = 'claude-haiku-4-5'

##########################
## PUll data 

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
question_df.groupby(['has_type2_question','qa_split','turn_type'])['question_id'].size().reset_index(name='count')



## Data structure

# class ConvFinQARecord(BaseModel):
#     report_id: str = Field(description="The id of the record")
#     doc: Document = Field(description="The document")
#     dialogue: Dialogue = Field(description="The conversational dialogue")
#     features: Features = Field(description="The features of the record, created by Tomoro to help you understand the data")


class ConvFinQAQuestion(BaseModel):
    report_id: str = Field(description="The id of the financial document for which question asked")
    question_id : str = Field(description="The id of the question unique for report & questions")
    conv_questions: str = Field(description="question submitted by user for report  '")

class ConvFinQAAnswer(ConvFinQAQuestion):
    conv_answers: str = Field(description="this is value that is numeric  or yes|no. if the question asks for percentage change use % format (e.g. 14.1% should be returned as 0.141) ")
    turn_program: str = Field(
        description='''
            this field icapture value or each sequential operations used to arrive at the final answer.             It is either numeric value same as conv_questions or the steps to arrive at the answer.
            
            For sequencial operations should list every operation(arg1, arg2)  in order, and be a numeric value if from be #0, #1 refering to a prior operation. 
            Output ONLY the program in this format: operation1(arg1, arg2), operation2(#0, arg3)

            Example of turn_program values:
             - No calcuation value pulled from financial document: 3.2 
             - Single-step: subtract(100, 50)
             - Multi-step: subtract(2010, 2009), multiply(203, 100)
             - Multi-step with reference: add(18.1, -6.3), subtract(#0, 9.4)
             - Multi-step: multiply(1.25, const_1000), divide(707, #0)

            Use #0, #1, etc to reference results from previous operations in the SAME program.
            Extract numbers directly from the context.

            Available operations:
            - add(num1, num2): Addition
            - subtract(num1, num2): Subtraction  
            - multiply(num1, num2): Multiplication
            - divide(num1, num2): Division
            - exp(base, power): Exponentiation
            - greater(num1, num2): Comparison

        ''')
    executed_answers: float | str = Field( description="Either numeric value to 5 decimal places or yes|no answer")



# Define a function to validate user input
def validate_ConvFinQAAnswer(output: str):
    """Validate output from the model fits the expected schema."""
    try:
        user_input = (
            ConvFinQAAnswer.model_validate_json(output)
        )
        print("output validated...")
        return user_input
    except Exception as e:
        print(f" Unexpected error: {e}")
        return None




# class OperationType(str, Enum):
#     """DSL operation types for ConvFinQA"""
#     ADD = "add"
#     SUBTRACT = "subtract"
#     MULTIPLY = "multiply"
#     DIVIDE = "divide"
#     EXP = "exp"
#     GREATER = "greater"
    



### AGENT sclass here
anthropic = Anthropic()
client = anthropic.Anthropic()


### Agent sctructure 
#  Planner - is the question require Number or Program
## retreive_data - pull the value from the document
## Program - 
## Tools use operations to answer the question
# if Program execute the program and return the answer.

######################

golden_examples = f'''
Example 1:  Calculation question with multi-step calculation with reference to prior step:
Question: "what is the percentage change in the unamortized debt issuance costs associated with its credit facilities from 2016 to 2017?",
conv_answers: "37.5%"
turn_program: "subtract(11, 8), divide(#0, 8)"
executed_answers: 0.375

Example 2: Yes/No question with single step calculation:
Question: "is this value greater than the cost of the early extinguishment?"
conv_answers: "yes",
turn_program: "multiply(400, 6.65%), greater(#0, 5)"
executed_answers: "yes

'''

assistant_header = f"""
You are a helpful financial assistant who will 
* used the provided financial document and answer user questions.  
* When there are multiple questions use the diaglogue history to answer the question.
* Say "I don't know" if you cannot answer the question with the provided document and dialogue history.

Examples of Question and Answers:
{golden_examples}

ALWAYS use tool provided "ConvFinQAAnswer" to structure your answer in the required format.

The questions related to eachother, use the history dialoge for context if you don't know what the input number required

You have access to the following financial document to answer the question: 



""" 

available_tools = []

tool_answer_structure = {
    "name": "ConvFinQAAnswer",
    "description": "Extract structured ConvFinQA answer data",
    "input_schema": ConvFinQAAnswer.model_json_schema()
}
available_tools.append(tool_answer_structure)


######################

#######################
## loopo through Document then Questions and Answer  

np.random.seed(42)
sample_n = 1000
reports_score = np.random.choice(question_df.query('data_key=="train"').report_id.unique(), size=sample_n, replace=False)

train_set = question_df.query(f'report_id in {list(reports_score)}')
train_set.groupby(['has_type2_question','turn_type'])['question_id'].size().reset_index(name='count')


# agent_answers = [] 

if os.path.exists(f"evaluations/{agent_name}_scoring.jsonl"):
    with open(f"evaluations/{agent_name}_scoring.jsonl", "r") as f:
        agent_answers = [json.loads(line) for line in f]
else:
    agent_answers = []
    with open(f"evaluations/{agent_name}_scoring.jsonl", "w") as f:
        for entry in agent_answers:
            f.write(json.dumps(entry) + "\n")


print('agent_answers loaded:', len(agent_answers))

# report_id = question_df.id.unique()[0]
for report_id in reports_score : 
    print('-'*50)
    print('Processing report_id:', report_id)
    #
    ## pull data
    query_data = [x for x in data.get('train') if x['id'] == report_id]
    if len(query_data) == 0:
        print(f"No data found for report_id: {report_id}")
    else :
        print('Found data for report_id:', report_id)
        query_data = query_data[0]
        questions = question_df.query(f'report_id == "{report_id}"')
    # structure the prompt for the model
    assistant_content_report = assistant_header +  f"""
    Fincial Document:
    {query_data['doc']}

    Report_id: 
    {query_data['id']}
    """
    messages = [{'role':'assistant', 'content':assistant_content_report}]
    # q = questions.to_dict(orient='records')[0]
    # q = questions.to_dict(orient='records')[1]
    # q = questions.to_dict(orient='records')[2]
    for q in questions.to_dict(orient='records'):
        if any(a['question_id'] == q['question_id'] for a in agent_answers):
            print(f"Skipping question_id {q['question_id']} as it has already been answered.")
            continue
        print(pd.Series(q.get('conv_questions')))
        messages.append({'role':'user', 'content':f''' 
            Question: {q.get('conv_questions')}
            Question_id: {q.get('question_id')}'''
            })
        ## single call to the model to get the answer for each question
        print('run model...')
        response = client.messages.create(max_tokens = 2024,
            model = 'claude-haiku-4-5', 
            messages = messages,
            tools =[tool_answer_structure],
            tool_choice={"type": "tool", "name": "ConvFinQAAnswer"},
            )
        # 
        print('tools ')
        tool_use = next(b for b in response.content if b.type == "tool_use" and b.name == 'ConvFinQAAnswer')
        result = ConvFinQAAnswer(**tool_use.input)
        agent_answers.append(result.model_dump())
        messages.append({'role':'assistant', 'content':result.model_dump_json()})
        with open(f"evaluations/{agent_name}_scoring.jsonl", "a") as f:
            f.write(json.dumps(result.model_dump()) + "\n")



#######################
## EVALUATION

with open(f"evaluations/{agent_name}_scoring.jsonl", "r") as f:
    answer_df = pd.DataFrame([json.loads(line) for line in f]) 

### Build evaluation set and score
## collate agent answers 
answer_df =answer_df.rename(columns={'conv_answers': f'conv_answers_{agent_name}', 'turn_program': f'turn_program_{agent_name}', 'executed_answers': f'executed_answers_{agent_name}'})
## Create master dataset with agent answers 
eval_df = question_df.copy() 
eval_df = eval_df.merge(answer_df,on='question_id', how='left'  )
eval_df['agent_answered'] = eval_df[f'conv_answers_{agent_name}'].notnull().apply(lambda x: 'Answered' if x else 'Not Answered')
## score 
pred_cols = ['executed_answers','turn_program']

## Predicted vs actual
eval_df[f'executed_answers_{agent_name}_score'] = (
     pd.to_numeric(eval_df['executed_answers'], errors='coerce').round(3) ==
 pd.to_numeric(eval_df[f'executed_answers_{agent_name}'], errors='coerce').round(3)).astype(int)
eval_df[f'turn_program_{agent_name}_score'] = (eval_df['turn_program'] == eval_df[f'turn_program_{agent_name}']).astype(int)

# for col in pred_cols:
#     eval_df[f'{col}_{agent_name}_score'] = (np.round(eval_df[col],3) == np.round(eval_df[f'{col}_{agent_name}'],3)).astype(int)

### TABLE 4/5 from the paper to compare against agent 
with open(f"evaluations/paper_performance.jsonl", "r") as f:
    performace_bar = pd.DataFrame([json.loads(line) for line in f]) 

### Agent performance  
eval_metrics = {
    'question_id':np.size,
    'executed_answers_'+agent_name+'_score':'mean',
    'turn_program_'+agent_name+'_score':'mean',
    }

eval_summ = eval_df.groupby(['agent_answered']).agg(eval_metrics).reset_index().rename(columns={'agent_answered': 'Methods'})
for attr in ['turn_type','has_type2_question','q_order']:
    print(attr)
    attr_summ = eval_df.groupby(['agent_answered',attr]).agg(eval_metrics).loc['Answered'].reset_index().rename(columns={attr: 'Methods'})
    attr_summ['Methods'] = attr + '-'+attr_summ['Methods'].astype(str) 
    eval_summ = pd.concat([eval_summ, attr_summ], ignore_index=True)


eval_summ.to_csv(f'evaluations/{agent_name}_final_summary.csv', index=False)
## comparison with paper performance
final_summ = performace_bar.merge(eval_summ, on=['Methods'], how='left').fillna(0)

final_summ.plot(x='Methods', y=['executed_answers_score',f'executed_answers_{agent_name}_score'], kind='bar', title='Executed Answers Score Comparison', legend=True)
plt.show() 

order_cols = sorted([f'{x}_score' for x in pred_cols] + [f'{x}_{agent_name}_score' for x in pred_cols])
final_summ[['Methods'] + order_cols]



###############
## Find test cases to improve agent for v2 

eval_df.query(f'agent_answered == "Answered" and turn_type == "Program" and executed_answers_agent_v1_score == 0 '
)[['question_id','conv_questions_x','executed_answers', f'executed_answers_{agent_name}','turn_program', f'turn_program_{agent_name}']]


eval_df.query(f'agent_answered == "Answered" and turn_type == "Program" and turn_program_agent_v1_score == 0 '
)[['question_id','conv_questions_x','turn_program', f'turn_program_{agent_name}']]

eval_df.query(f'agent_answered == "Answered"').to_csv('evaluations/agent_v1_detailed_results.csv', index=False)


