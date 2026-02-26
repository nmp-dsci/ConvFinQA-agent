
from pydantic import BaseModel, Field, validator, ValidationError
import pandas as pd 
import numpy as np 
import anthropic 
import os,json, re 
from dotenv import load_dotenv

load_dotenv()
agent_name = 'agent_v3'




## Data structure

class ConvFinQAQuestion(BaseModel):
    report_id: str = Field(description="The id of the financial document for which question asked")
    question_id : str = Field(description="The id of the question unique for report & questions")
    conv_questions: str = Field(description="question submitted by user for report  '")
    turn_type: str = Field(description="Either Number or Program.  Number retrieved directly from financial document, Program retrieves new numbers from financial report or answers from prior questions in the dialogue. ")
    qa_split: bool = Field(description="Separtes if it is question that is Simple (false) and a value that can be retrieve (Number) or calculated (Program) from previous answers retrived  or Complex (true) where need to augument a previous question for a different year for example and either retrive or calculate the answer. ")

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
             - Multi-step with reference previous calculation as #0: add(18.1, -6.3), subtract(#0, 9.4)
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


######################

golden_examples = f'''
Example 1:  Calculation question with multi-step calculation with reference to prior step:
Question: "what is the percentage change in the unamortized debt issuance costs associated with its credit facilities from 2016 to 2017?",
conv_answers: "37.5%"
turn_program: "subtract(11, 8), divide(#0, 8)"
executed_answers: 0.375
turn_type: "Program"
qa_split: False 

Example 2: Yes/No question with single step calculation:
Question: "is this value greater than the cost of the early extinguishment?"
conv_answers: "yes",
turn_program: "multiply(400, 6.65%), greater(#0, 5)"
executed_answers: "yes
turn_type: "Program"
qa_split: False 

Example 3: summing multiple prior answers in dialogue 
Question: "what is the total sum of square feet owned?"
conv_answers: "377000",
turn_program: "add(160000, 80000), add(#0, 70000), add(#1, 67000)"
executed_answers: 377000
turn_type: "Program"
qa_split: False 

Example 4:  Question that is a Number retrival
Question: "what is the balance of cash and cash equivalents at the end of 2009?"
conv_answers: "364221",
turn_program: "364221"
executed_answers: 364221
turn_type: "Number"
qa_split: False 

Example 5: Question that is a Number retrival, need to get previous question and retrieve data for a different year 
Question: "what about in 2017?"
conv_answers: "85.98",
turn_program: "85.98"
executed_answers: 85.98
turn_type: "Number"
qa_split: True 

Example 6: qa_split = True, need to retrieve data augmenting previous questions  and run a calcuation 
Question: "and for the two years prior to 2008, what was the total recognized expense related to defined contribution plans?" 
conv_answers: "110.2",
turn_program: "add(61.9, 48.3)"
executed_answers: 110.2
turn_type: "Program"
qa_split: True 

Example 7: qa_split = True, need to look at previous question and answers and add to it based on the context of this questions 
Question: "including 2008, what then becomes that total?" 
conv_answers: "155.7",
turn_program: "add(61.9, 48.3), add(#0, 45.5)"
executed_answers: 155.7
turn_type: "Program"
qa_split: True 

'''

assistant_header = f"""
You are a helpful financial assistant who will answer questions that are either number or yes/no retrieval or complex calcuation. 

* Use the provided financial document and answer user questions. 
* The questions related to eachother, use the history dialoge for context if you don't know what the input number required. 


The plan to answer the question is as follows: 
1. Determine if `turn_type` is "Number" or "Program" based on the question. and answer accordingly.
    * Number is retrieved directly from the document. examples
        a. what is the total revenue in 2019? 
        b. what is total net assets in 2008? 
    * Program require more steps: 
        a. what numbers are needed, are the from the financial report or prior question in the dialogue. 
        b. what operations are needed to arrive at the answer. the operations can be add, subtract, multiply, divide, exp, greater.
        c. with the numbers and the operations. Calculate the answer 
2. Determine the classification of qa_split: 
    * True: means the question is a augmentation of prior quetions requiring agent to change prior question and retrieve data and and if a turn_type == "Program" then run the calculation with the retrived data 
    * False: means the question can be answer retrieve data based on the question or leveraging prior answers to questions to do a calcuation (Program) 
3. Retrieve the data. Uset the question or create multiple new questions to be used to retrieve data from the financial document. 
4. If turn_type is: 
    * Number: then return the answer based on the question and retrived data.
    * Program: then determine the operations needed to arrive at the answer and execute the program to arrive at the final answer.
5. If turn_type = "Program" run the calcuation to arrive at the final answer.

Examples of Question and Answers:
{golden_examples}

ALWAYS use tool provided "ConvFinQAAnswer" to structure your answer in the required format.


You have access to the following financial document to answer the question: 

""" 



# test report_id = 'Double_GIS/2008/page_83.pdf'
def report_data(report_id):
    '''Pull data for provided report_id''' 
    with open( "data/convfinqa_dataset.json") as f:
        data = json.load(f) 

    data = data.get('train')+ data.get('dev')
    if report_id is None: 
        return "report_id is required for Agent to answer question" 
    else: 
        report_data = [x for x in data if x['id'] == report_id]
        if len(report_data) == 0:
            return f"No data found for report_id: {report_id}"
        else :
            # print('Found Report, ask questions ', report_id)
            report_data = report_data[0].get('doc')
    return report_data


class recordQAAgent:
    def __init__(self, agent_name=agent_name):
        self.agent_name = agent_name
        self.available_prompts = []
        self.client = anthropic.Anthropic()
        self.model_name = 'claude-haiku-4-5'
        self.available_tools = []
        self.quesion = {}
        self.messages = []
        self.output_template = 'ConvFinQAAnswer'
        self.tool_answer_structure = {
            "name": "ConvFinQAAnswer",
            "description": "Extract structured ConvFinQA answer data",
            "input_schema": ConvFinQAAnswer.model_json_schema()
        }
        self.available_tools.append(self.tool_answer_structure)
    
    def _init_record(self,report_id):
        self.report_id = report_id
        self.report_doc = report_data(self.report_id)
        ## clear messages history for new Q&A  
        self.messages = []
        self.question_n = 0
        assistant_content_report = assistant_header +  f"""
        Fincial Document:
        {self.report_doc}
        Report_id: 
        {self.report_id}
        """
        self.messages.append({'role':'assistant', 'content':assistant_content_report})
   
    def query_agent1(self, question_raw):
        agent_step = f'{self.agent_name}_i1'
        self.question = {'conv_questions': question_raw, 'question_id': f'{self.report_id}_q{self.question_n}'}
        self.messages.append({'role':'user', 'content':f''' 
            Question: {self.question.get('conv_questions')}
            Question_id: {self.question.get('question_id')}''' })
        agent1_response = self.client.messages.create(max_tokens = 2024,
            model = 'claude-haiku-4-5', 
            messages = self.messages,
            tools =self.available_tools,
            tool_choice={"type": "tool", "name": self.output_template },
            )
        # dump output 
        tool_use = next(b for b in agent1_response.content if b.type == "tool_use" and b.name == self.output_template )
        result = globals().get(self.output_template )(**tool_use.input)
        self.messages.append({'role':'assistant', 'content':result.model_dump_json()})
        self.question_n += 1
        return result.model_dump()
    
    def query_agent2(self):
        agent_step = f'{self.agent_name}_i2'
        self.messages.append({'role':'user', 'content':f''' 
            Based on Question and Output: {self.messages[-1]}   

            Validate if the answer is correct and if not updated provide the correct answer.  
            Review the answer Output and recalcuate the answer if needed based on the question, financial report and message history 

            Steps to follow for question review
            1. Review turn_type classification. 
                * "Number" then turn_program should be a single numeric value 
                * "Program" the validate turn_program is should be a sequence of opterations a value is directly from Financil Report or #0, #1, etc to reference the operation preceding it in the same string 
            2. Review qa_split:
                * True means that is multi-hop question that requires retrieving data for different variations of prior questions and returning value or running calculations based on that.
                * False means the question can be answer retrieve data based on the question or leveraging prior answers
            3. Review of turn_program:  review the turn_program in accordance format defined in ConvFinQAAnswer 
            4. Recalulate any of the outputs to question to questions based on the above review and update the answer
                * executed_answers
                * turn_program: pay strict attention to format, use interger over decimal. Consistent decimal place. All values either from the report or are a operation and prior operation results refered too as #0, #1, etc.
                * turn_type: update this to Number or Program, especially if turn_progam is numeric value then this should be number
                * qa_split: update True or False, True if its an augmentation of prior questions in message dialogue  to retrieve data 
            ''' })        
        iter2_response = self.client.messages.create(max_tokens = 2024,
            model = 'claude-haiku-4-5', 
            messages = self.messages,
            tools =[self.tool_answer_structure],
            tool_choice={"type": "tool", "name":self.output_template },
        )        
        # dump output 
        tool_use = next(b for b in iter2_response.content if b.type == "tool_use" and b.name == self.output_template )
        result = globals().get(self.output_template ) (**tool_use.input)
        self.messages.append({'role':'assistant', 'content':result.model_dump_json()})
        return result.model_dump()



