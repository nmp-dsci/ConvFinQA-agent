  """                                                                           
  Step 1: Data Scope                                                            
  ------------------                                                            
  * goal is to build an understanding of: 
     --type of questions ( numeric vs complex)   
     --format to inform LLM /pydantic schema design                 
  """

## root is the root directory of the project
import json
from pathlib import Path
import pandas as pd 
import matplotlib.pyplot as plt
import re 

with open( "data/convfinqa_dataset.json") as f:
    data = json.load(f)



##  train  3037,  dev   421
pd.DataFrame([{
    'key': x, 
    'len': len(data[x]),
} for x in data.keys() ])


## check if keys exist for all : all match
key_df = pd.DataFrame([x.keys() for i,x in enumerate(data['dev'])]) 
assert key_df.isnull().sum(axis=0).sum() == 0 , 'missing values'

## structure of the data 
for key in data.keys():  
    globals()[f'{key}_df'] = pd.concat([pd.DataFrame({**x['dialogue'], 'data_key': key, 'id': x['id'],'q_order':range(len(x['dialogue']['conv_questions']))}) for x in data[key ]], ignore_index=True)

data_df = pd.concat([globals()[f'{key}_df'] for key in data.keys() ],ignore_index=True) 
data_df['turn_type'] = pd.to_numeric(data_df['turn_program'],errors='coerce').apply(lambda x: 'numeric' if pd.notnull(x) else 'calculation')
## decompossing attributes of `turn_program` to understand the complexity/scope  of the calculations
data_df['turn_program_actions'] = data_df['turn_program'].str.split('(?<=\)),')
data_df['turn_program_actions_n']  = data_df['turn_program_actions'].apply(len)
data_df['turn_program_calcs'] = data_df['turn_program_actions'].apply(lambda x: [ m.group(1) if (m := re.match(r'\s*(\w+)\(', s)) else None for s in x ])

### TOP types of actions 
data_df['turn_program_calcs'].explode().value_counts()

'''
data_df['turn_program_calcs'].explode().value_counts()
turn_program_calcs
subtract    5131
divide      4280
add         2457
multiply     894
greater       40
exp            4
'''

### 
data_df['executed_answers_num'] = pd.to_numeric(data_df['executed_answers'],errors='coerce')
.notnull().sum() / len(data_df)
data_df.query('executed_answers_num!= executed_answers_num') 
 




[re.match(r'(\w+)\(', s).group(1) for s in strings]

pd.to_numeric(data_df.query('turn_type == "calculation" and q_order == 0')['turn_program'], errors='coerce').head(10)

##  View Calculation vs Numeric distribution across data keys and question order
counts = data_df.groupby(['data_key','q_order','turn_type']).size().unstack('turn_type')
pct = counts.div(counts.sum(axis=1), axis=0).mul(100).round(1)             
print(pct) 
pct = pct.sort_index(level=1)

pct.plot(kind='bar', stacked=True, figsize=(10,6), colormap='viridis')
plt.show() 


data_df.query('turn_type == "calculation" and q_order == 0').head(10)

data_df.loc[12]
data_df.loc[12,'conv_questions']

######  View distribution of calculation vs numeric across data keys and question order

dist = data_df.groupby(['data_key','q_order','turn_type','turn_program_actions_n']).size().unstack('data_key')
dist = dist/ dist.sum(axis=0)

dist.plot(kind='bar', stacked=False , figsize=(10,6), colormap='viridis')
plt.show() 

dist = data_df.groupby(['data_key','q_order','turn_type']).size().unstack('data_key')
dist = dist/ dist.sum(axis=0)

dist.plot(kind='bar', stacked=False , figsize=(10,6), colormap='viridis')
plt.show() 


dist = data_df.groupby(['data_key','turn_type','turn_program_actions_n']).size().unstack('data_key')
dist = dist/ dist.sum(axis=0)

dist.plot(kind='bar', stacked=False , figsize=(10,6), colormap='viridis')
plt.show() 


