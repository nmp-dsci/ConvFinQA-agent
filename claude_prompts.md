

## Get claude to add add 100 evals 

I'm running promptfoo to evalutaion of AI agent. using "data/convfinqa_dataset.json", under "train" randomly pull 100 records and add "id" / "doc" as a new json in tests/doc#.json where # is the test number. 

Correspondingly add to tests/multiturn_tests.yaml add the "dialogue/conv_questions" and "dialogue/conv_answers" using the same format that already exists in that file. 

The tests/doc#.json created to be linked to the questions in the multiturn_test.yaml so that it's testing the right questions against document 



## Get claude to make v4 version 

can you update provider_v4.py and turn it into a multi agent process   
  1. Agent 1: first agent that receives the question and plan out how to answer the question, leveraging the other agents. It doesn't need tools because it classifies the job into numeric or calculation and whether the values to pull are from document or derivative of previous questions and then retieve numeric values and finally do calcuation.                             
  2. Agent 2: agent that retrives numeric values from document, doesn't rely of previous user questions for context                         
  3. Agent 3:  agent needs to retrieve numeric values from document that are based on users previous questions to know values to pull. 
  4. Agent 4: calculation agent, this agent takes retreived from Agent 2 /3 and does the calculation asked for from the question. 
  4. Agent 5: reviewer agent, with the final answer this agent makes sure is a number or percentage with not currency symbols or commas.     

## claude prompt v5 

can up make a plan for a multi-agent orchestrator agent that can answer multiturn questions on financial documents. Update `provider_v5.py` agent and its current implementation of promptfoo runnable and MCP. 


Route incoming question comes in the "planner agent" that categorises type of question before answering: 
 * `turn_type`: either "Number" from document or "Program" that required multiple "Number"s from document or previous assistant answers for a calculation like percentage or growth.  
 * `type2_question`: Type 2 question's related to recalculating previous question / answers. So to answer question need to look into instory

 based on the `turn_type` and `type2_question`: 
  * if `type2_question` is True then send request off to type2_sub_agent than takes question and message history and generates all Number questions to retrieve from  document and rturn to "planner agent" . 

  * if `type2_question` is False then planner agent can create the Number questions to retrieve from financial document itself. 
  * Now it can send financial document Number questions to document_sub_agent that takes a list of questions and retrieves the Number from the document. 
  * if `turn_type` equal Number it will be a single question retrieved from financial document with  calculation required. 
  * if `turn_type` is "Program" there will be multiple numbers retrieved from data. and the Planner agent needs to generate the sequential series of cacluations required to answer the question basead on the Numbers retrieved from financial document or message history. Then hand over to a program_sub_agent than has access to MCP tool `server_calculator.py` to run calculations on numbers to answer question. 
  * Final `review_sub_agent` will review the final answer and make sure its a number or percentage with no commas / currency symbols. Do not provide explanation. 

<turn_type>
examples: 
* Number is retrieved directly from the document. examples
    a. what is the total revenue in 2019? 
    b. what is total net assets in 2008? 
* Program require more steps: 
    a. what numbers are needed, are the from the financial report or prior question in the dialogue. 
    b. what operations are needed to arrive at the answer. the operations can be add, subtract, multiply, divide, exp, greater.
    c. with the numbers and the operations. Calculate the answer 
<turn_type>



<type2_question>
examples
 * "and over the subsequent year of that period, what was that change in the rental expense?" relies on previous question  "from 2018 to 2019, what was the change in the total rental expense under operating leases?" to calculated for 2019 to 2020 change in total rental expense under operating lease
 * "and in the last year of that period, what was the total amount spent on the repurchase of shares?" relies on previous question "what was the change in the average price paid per share from 2012 to 2013?" to under
</type2_question>




