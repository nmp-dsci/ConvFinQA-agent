'''

Test LLM agent v2: plan / multi-agent 
## Agent1: Plan & Answer   
## Reflection: on answer and improve  
## Tools: to control output 
## Chain of though / golden expands 


'''


from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from contextlib import AsyncExitStack
import json
import asyncio
import nest_asyncio


from pydantic import BaseModel, Field, validator, ValidationError
import pandas as pd 
import numpy as np 
import anthropic 
import os,json, re ,sys
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from agent_utils import training_data,load_agent_answers
from agent_v3 import * 



pd.options.display.max_columns  = 100

load_dotenv()


agent_name = 'agent_v3'

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




# def agent_v2_backtest(sample_n=10 , seed=42 ):
#     """
#     Backtest the agent  for Evaluation of performance. 

#     """
# ##Define data set to score 
# with open( "data/convfinqa_dataset.json") as f:
#     data = json.load(f)
# question_df = training_data()
# ## pull records to score
# np.random.seed(seed)
# reports_score = np.random.choice(question_df.query('data_key=="train"').report_id.unique(), size=sample_n, replace=False)
# train_set = question_df.query(f'report_id in {list(reports_score)}')
# train_set.groupby(['has_type2_question','turn_type'])['question_id'].size().reset_index(name='count')
# ## Init Agent
# record_agent = recordQAAgent()


# # report_id = 'Double_GIS/2008/page_83.pdf'
# for report_id in reports_score : 
#     print('-'*50)
#     print('Processing report_id:', report_id)
#     record_agent._init_record(report_id) 
#     ## pull data
#     query_data = [x for x in data.get('train') if x['id'] == report_id]
#     query_data = query_data[0]
#     questions = question_df.query(f'report_id == "{report_id}"')
#     # q = questions.conv_questions.to_list()[0]
#     for q in questions.conv_questions.to_list():
#         #### Agent 1: Planner and Execute
#         agent_step = f'{agent_name}_i1'
#         qid = f'{record_agent.report_id}_q{record_agent.question_n}'
#         agent_answers = load_agent_answers(agent_name=f'{agent_step}')
#         if any(a['question_id'] == qid for a in agent_answers):
#             print(f"ITER1:Report & Question already answered")
#             continue
#         result = record_agent.query_agent1(q) 
#         with open(f"evaluations/{agent_step}_scoring.jsonl", "a") as f:
#             f.write(json.dumps(result) + "\n")
#         #### Agent 2: Review and Validation
#         agent_step = f'{agent_name}_i2'
#         agent_answers = load_agent_answers(agent_name=f'{agent_step}')
#         if any(a['question_id'] == qid for a in agent_answers):
#             print(f"ITER2:Report & Question already answered")
#             continue     
#         result = record_agent.query_agent2() 
#         with open(f"evaluations/{agent_step}_scoring.jsonl", "a") as f:
#             f.write(json.dumps(result) + "\n")       


## Agent class init


class  MCP_ChatBot:
    def __init__(self):
        self.agent_name = agent_name
        self.client = anthropic.Anthropic()
        self.model_name = 'claude-haiku-4-5'
        self.question = {}
        self.messages = []
        self.output_template = 'ConvFinQAAnswer'
        self.tool_answer_structure = {
            "name": "ConvFinQAAnswer",
            "description": "Extract structured ConvFinQA answer data",
            "input_schema": ConvFinQAAnswer.model_json_schema()
        }
        ## MCP 
        self.exit_stack = AsyncExitStack()
        self.sessions = {}
        self.available_tools = []
        self.available_prompts = []
    #
    async def connect_mcp_server(self,server_name, server_config):
        try:
            # Define server parameters
            server_params = StdioServerParameters(**server_config)
            # Connect to the 
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            session = await self.exit_stack.enter_async_context( ClientSession(read, write))
            await session.initialize() 
            # add tools / prompts / resources
            try:
                # List tools
                response = await session.list_tools()
                for tool in response.tools:
                    self.sessions[tool.name] = session
                    self.available_tools.append({
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.inputSchema
                    })
                # List prompts
                prompts_response = await session.list_prompts()
                if prompts_response and prompts_response.prompts:
                    for prompt in prompts_response.prompts:
                        self.sessions[prompt.name] = session
                        self.available_prompts.append({
                            "name": prompt.name,
                            "description": prompt.description,
                            "arguments": prompt.arguments
                        })
                # List resources
                resources_response = await session.list_resources()
                if resources_response and resources_response.resources:
                    for resource in resources_response.resources:
                        resource_uri = str(resource.uri)
                        self.sessions[resource_uri] = session
                #
            except Exception as e:
                print(f"Error {e}")
            #
        except Exception as e:
            print(f"Error connecting to {server_name}: {e}")
    #
    async def connect_mcp_servers(self):
        try:
            with open("mcp/server_config.json", "r") as file:
                mcp_data = json.load(file)
            servers = mcp_data.get("mcpServers", {})
            for server_name, server_config in servers.items():
                await self.connect_mcp_server(server_name, server_config)
        except Exception as e:
            print(f"Error loading server config: {e}")
            raise


    async def process_query(self, query):
        self.messages.append({'role':'user', 'content':query}) 
        
        while True:
            response = self.client.messages.create(
                max_tokens = 2024,
                model = self.model_name, 
                tools = self.available_tools,
                messages = self.messages
            )
            
            assistant_content = []
            has_tool_use = False
            
            # print('*' * 50 )
            # print(response.content) 
            # print('*' * 50 )

            for content in response.content:
                if content.type == 'text':
                    print(content.text)
                    assistant_content.append(content)
                elif content.type == 'tool_use':
                    has_tool_use = True
                    assistant_content.append(content)
                    self.messages.append({'role':'assistant', 'content':assistant_content})

                    print('*' * 50 )
                    print(content)             
                    # Get session and call tool
                    session = self.sessions.get(content.name)
                    if not session:
                        print(f"Tool '{content.name}' not found.")
                        break
                        

                    result = await session.call_tool(content.name, arguments=content.input)

                    print(result) 
                    print('*' * 50 )

                    self.messages.append({
                        "role": "user", 
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_use_id": content.id,
                                "content": result.content
                            }
                        ]
                    })
            
            # Exit loop if no tool was used
            if not has_tool_use:
                break

    
    async def cleanup(self):
        await self.exit_stack.aclose()

            

    async def chat_loop(self):
        print("\nMCP Chatbot Started!")
        print("Type your queries or 'quit' to exit.")
        print("Use @folders to see available topics")
        print("Use @<topic> to search papers in that topic")
        print("Use /prompts to list available prompts")
        print("Use /prompt <name> <arg1=value1> to execute a prompt")
        
        while True:
            try:
                self.query = input("\nQuery: ").strip()
                if not self.query:
                    continue
        
                if self.query.lower() == 'quit':
                    break
                
                # Check for @resource syntax first
                if self.query.startswith('@'):
                    # Remove @ sign  
                    topic = query[1:]
                    if topic == "folders":
                        resource_uri = "papers://folders"
                    else:
                        resource_uri = f"papers://{topic}"
                    await self.get_resource(resource_uri)
                    continue
                
                # Check for /command syntax
                if self.query.startswith('/'):
                    parts = self.query.split()
                    command = parts[0].lower()
                    
                    if command == '/prompts':
                        await self.list_prompts()
                    elif command == '/prompt':
                        if len(parts) < 2:
                            print("Usage: /prompt <name> <arg1=value1> <arg2=value2>")
                            continue
                        
                        prompt_name = parts[1]
                        args = {}
                        
                        # Parse arguments
                        for arg in parts[2:]:
                            if '=' in arg:
                                key, value = arg.split('=', 1)
                                args[key] = value
                        
                        await self.execute_prompt(prompt_name, args)
                    else:
                        print(f"Unknown command: {command}")
                    continue

                else: 
                    self.messages = []
                    await self.process_query(self.query)
                    
            except Exception as e:
                print(f"\nError: {str(e)}")
    
    async def cleanup(self):
        await self.exit_stack.aclose()



async def main():
    chatbot = MCP_ChatBot()
    try:
        await chatbot.connect_mcp_servers()
        await chatbot.chat_loop()
        # await chatbot.process_query("what is (5 + 5  * 10) / 8 - 10?")
        # await chatbot.process_query("what is (5 + 5  * 10) / 8 greater than (5 + 3  * 10) / 8 ?")
    
    finally:
        await chatbot.cleanup()




if __name__ == "__main__":
    asyncio.run(main())


