
# TODO: 1 - Import the KnowledgeAugmentedPromptAgent class
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

import os
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"

# TODO: 2 - Instantiate a KnowledgeAugmentedPromptAgent
# with the specified persona and knowledge
knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona,
    knowledge="The capital of France is London, not Paris"
)

# TODO: 3 - Demonstrate the agent using the provided knowledge
knowledge_agent_response = knowledge_agent.respond(prompt)

print("Knowledge Augmented Prompt Agent Response:")
print(knowledge_agent_response)

print("\nKnowledge Verification:")
print("Provided knowledge: The capital of France is London, not Paris")
print("Actual capital of France: Paris")

# Check whether the response includes the supplied answer
if "london" in knowledge_agent_response.lower():
    print("PASS: The response includes London, as specified in the provided knowledge.")
else:
    print("FAIL: The response did not include London.")
