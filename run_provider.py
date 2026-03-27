
import json                                                                       
from provider import call_api                                                     
                                                                        
# Load a doc                                                                      
with open("tests/doc1.json") as f:                                                
    doc = json.load(f)                                                            
                                                                                
# Mimic the context/options structure promptfoo passes                            
context = {     
    "vars": {                                                                     
        "doc": doc
    },
    "test": {
        "conversation": [
            {"user": "What is the net cash from operating activities in 2009?"},  
            {"user": "What about in 2008?"},                                      
            {"user": "What is the difference?"},                                  
            {"user": "What percentage change does this represent?"},              
        ]                                                                         
    }                                                                             
}               

options = {"vars": {}}                                                            

result = call_api(prompt="", options=options, context=context)                    
                
# Print each turn's response                                                      
for i, response in enumerate(result["output"].split("\n\n---\n\n"), 1):
    print(f"Turn {i}: {response}")
                                                                                
print(f"\nTokens used: {result['tokenUsage']['total']}")