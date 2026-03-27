
golden_examples = f'''
    Example 1:  Calculation question with multi-step calculation with reference to prior step:
    Question: "what is the percentage change in the unamortized debt issuance costs associated with its credit facilities from 2016 to 2017?",
    value: "37.5%"

    Example 2: Yes/No question with single step calculation:
    Question: "is this value greater than the cost of the early extinguishment?"
    value: "yes",

    Example 3: summing multiple prior answers in dialogue 
    Question: "what is the total sum of square feet owned?"
    value: "377000",

    Example 4:  Question that is a Number retrival
    Question: "what is the balance of cash and cash equivalents at the end of 2009?"
    value: "364221",

    Example 5: Question that is a Number retrival, need to get previous question and retrieve data for a different year 
    Question: "what about in 2017?"
    value: "85.98",

    Example 6: qa_split = True, need to retrieve data augmenting previous questions  and run a calcuation 
    Question: "and for the two years prior to 2008, what was the total recognized expense related to defined contribution plans?" 
    value: "110.2",

    Example 7: qa_split = True, need to look at previous question and answers and add to it based on the context of this questions 
    Question: "including 2008, what then becomes that total?" 
    value: "155.7",

'''

chain_of_thought = '''

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
'''





