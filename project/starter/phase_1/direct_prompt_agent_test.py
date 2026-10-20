
#s Test script for DirectPromptAgent class

# TODO: 1 - Import the DirectPromptAgent class
from workflow_agents.base_agents import DirectPromptAgent

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# TODO: 2 - Load the OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

# TODO: 3 - Instantiate the DirectPromptAgent
direct_agent = DirectPromptAgent(openai_api_key)

# TODO: 4 - Send the prompt and store the response
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)

# TODO: 5 - Explain the agent's knowledge source
print(
    "The DirectPromptAgent generates its response using "
    "the pretrained knowledge of the GPT-3.5 Turbo model. "
    "It does not use external tools, retrieval systems, "
    "or additional context."
)
