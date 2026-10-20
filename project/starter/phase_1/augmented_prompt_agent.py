# TODO: 1 - Import the AugmentedPromptAgent class
from workflow_agents.base_agents import AugmentedPromptAgent

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# TODO: 2 - Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona
)

# TODO: 3 - Send the prompt to the agent and store the response
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# TODO: 4 - Explain the knowledge source and persona influence
# The agent likely used the pretrained knowledge of the GPT-3.5 Turbo
# language model to identify Paris as the capital of France.
# It did not use external knowledge sources, retrieval systems,
# or databases to generate its response.
#
# The system prompt instructed the agent to adopt the persona
# of a college professor. This influenced the response's tone,
# style, and structure, specifically instructing the agent
# to begin every answer with "Dear students,".
