"""
Main typer app for ConvFinQA
"""

import typer
from rich import print as rich_print
import json ,random 
from typing import Optional
import questionary
from agent_utils import training_data,load_agent_answers
from agent_v2 import * 


app = typer.Typer(
    name="main",
    help="Boilerplate app for ConvFinQA",
    add_completion=True,
    no_args_is_help=True,
)




# test report_id = 'Double_GIS/2008/page_83.pdf'
def get_report_data(report_id=None):
    '''Pull data for provided report_id''' 
    with open( "data/convfinqa_dataset.json") as f:
        data = json.load(f)    
    
    data = data.get('train')+ data.get('dev')
    if report_id is None: 
        return [x.get('id') for x in data]  
    else: 
        report_data = [x for x in data if x['id'] == report_id]
        if len(report_data) == 0:
            return [x.get('id') for x in data]
        else :
            return report_data[0]




def select_record(record_id):
    """Loop until a valid record_id is selected, returns record_id"""
    while True:
        if record_id  in {"exit", "quit"}:
            typer.echo("Goodbye!")
            raise typer.Exit()          # ✅ exit app entirely
        
        report_data = get_report_data(record_id)
        if not isinstance(report_data, list):
            rich_print(f"[green][bold]assistant:[/bold] Record found for '{record_id}'[/green]")
            return report_data
        else:
            rich_print(f"[yellow][bold]assistant:[/bold] Select Record (Autocomplete)[/yellow]")
            selection = questionary.autocomplete(
                "Select a record:",
                choices=["✏️  Enter Record_ID","exit","quit"] + report_data 
            ).ask()
            if selection == "✏️  Enter Record_ID":
                record_id = questionary.text("Enter record_id:").ask()
            else:
                record_id = selection





@app.command()
def chat(
    record_id: str  = typer.Argument(None, help="ID of the record to chat about"),
) -> None:
    """Select a report and ask questions about a specific record,"""
    record_agent = recordQAAgent()

    while True:

        # 1. SELECT VALID Rcord
        report_data = select_record(record_id)
        typer.echo(f"✅ SELECTED: {report_data.get('id')}")
        
        ## Load data for record / update LLM 
        record_agent._init_record(report_data.get('id'))
        report_qs  = report_data.get('dialogue').get('conv_questions')
        report_qs = [f'Q{i}-{x}' for i,x in enumerate(report_qs)]

        report_answers = '\n'.join([ f"[purple][bold]{x}[purple][bold]: [green][bold]{report_data.get('dialogue').get('executed_answers')[i]}[green][bold] " for i,x in enumerate(report_qs)])

        while True: 
            try:
                enter_question = questionary.autocomplete(
                    f"{report_data.get('id')}>>> ",
                    choices=["change","switch","exit","quit","✏️  New Question",] + report_qs 
                ).ask()

                if enter_question == "✏️  Enter Question":
                    question = questionary.text("Question:").ask()
                    llm_response = record_agent.query_agent1(enter_question)

                elif not enter_question:
                    continue

                elif enter_question.strip().lower() in {"answers"}: 
                    rich_print(report_answers)
                    continue

                elif enter_question.strip().lower() in {"exit", "quit"}:
                    typer.echo("Goodbye!")
                    raise typer.Exit()          # ✅ exit app entirely

                elif enter_question.strip().lower() in {"change", "switch"}:
                    record_id = None            # ✅ flip back to selection
                    break

                else:
                    llm_response = record_agent.query_agent1(enter_question)

  
                answer = llm_response.get('executed_answers')  if llm_response.get('executed_answers')  != "<UNKNOWN>" else f"Question not returning result: '{enter_question}' "
                rich_print(f"[blue][bold]Assistant:[/bold] {llm_response.get('executed_answers')}[/blue]")
            except typer.Exit:
                raise
            except Exception as e:
                print(f"\nError: {str(e)}")



if __name__ == "__main__":
    app()



'''
---- TEST AGENT --- 
record_agent = recordQAAgent()
record_agent._init_record('Single_AES/2017/page_110.pdf-3') 
'what is the value of debt obligations due in under 1 year?'

"what are total debt obligations?""20404"
"what is the value of debt obligations due in under 1 year?","2250",
"what is the value of total debt obligations less those due in under 1 year?""18154"
"what is that divided by total debt obligations?""89.0%"

'''