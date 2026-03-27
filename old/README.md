# ConvFinQA Assignment

### CLI App Structure



### Prerequisites
- Python 3.12+
- [UV environment manager](https://docs.astral.sh/uv/getting-started/installation/)

### Setup
1. Clone this repository
2. Use the UV environment manager to install dependencies:



### Run Chat App
Test out the Agent running app and selecting by Report and Question. 
```bash 

# set up env -- reinstall 
uv sync

# run app: (then > Select Record > Ask Question 
un run main 

```

#### `Select Record`
Search or autocomplete from available reports, press `ENTER` to confirm

[![Chat](figures/app_select_record.png)](figures/app_select_record.png)  


#### `Enter Question` 
Free form or autocomplete with ConFinQA questions

[![Chat](figures/app_enter_question.png)](figures/app_enter_question.png)  

See Record Question and Answers 

[![Chat](figures/app_see_answers.png)](figures/app_see_answers.png)


| Input | Action |
|---|---|
| Free text question | Query the selected record |
| Autocomplete (type `q` to start) | Browse training data questions |
| `answers` | To see all Question & Answers from ConFinQA data for record |
| `change` or `switch` | Go back to Record Selection |
| `quit` or `exit` | Shutdown app |

    





