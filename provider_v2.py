import json
import os
from anthropic import Anthropic
from dotenv import load_dotenv

from prompt_utils import *   

load_dotenv()
client = Anthropic()



def call_api(prompt, options, context):
    """
    When using conversation: test format, promptfoo calls this once with all turns
    defined in context.test.conversation. We iterate through all turns, maintaining
    conversation history, and return all responses so per-turn assertions can be evaluated.
    """

    # Load doc from context.vars (promptfoo resolves file:// refs and places them here)
    doc = context.get("vars", {}).get("doc", "")
    if isinstance(doc, str) and doc.startswith("{"):
        doc = json.loads(doc)

    # Build system prompt with document
    system_prompt = f"""You are a financial analyst assistant.
    Use the following document to answer all questions accurately.
    Maintain context from previous questions in the conversation.


    <document>
    {json.dumps(doc, indent=2)}
    </document>

    <examples>
    {golden_examples}
    </examples>

    <chain_of_thought> 
    {chain_of_thought}
    </chain_of_thought>

Only output the final answer value no explanation  """

    turns = context.get("test", {}).get("conversation", [])

    # If no conversation turns (single-turn usage), fall back to simple call
    if not turns:
        messages = [{"role": "user", "content": prompt}]
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
        )
        text = response.content[0].text
        return {
            "output": text,
            "tokenUsage": {
                "total": response.usage.input_tokens + response.usage.output_tokens,
                "prompt": response.usage.input_tokens,
                "completion": response.usage.output_tokens,
            },
        }

    # Multi-turn: run all conversation turns sequentially
    messages = []
    all_responses = []
    total_input_tokens = 0
    total_output_tokens = 0

    for turn in turns:
        user_message = turn["user"]
        messages.append({"role": "user", "content": user_message})

        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=1024,
            system=system_prompt,
            messages=messages,
            
        )

        assistant_text = response.content[0].text
        messages.append({"role": "assistant", "content": assistant_text})
        all_responses.append(assistant_text)
        total_input_tokens += response.usage.input_tokens
        total_output_tokens += response.usage.output_tokens

    # Return all responses joined so assertions on any turn can be evaluated
    combined_output = "\n\n---\n\n".join(all_responses)

    return {
        "output": combined_output,
        "tokenUsage": {
            "total": total_input_tokens + total_output_tokens,
            "prompt": total_input_tokens,
            "completion": total_output_tokens,
        },
    }
