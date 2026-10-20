
# agentic_workflow.py

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add the Phase 1 agent package directory so this script can run directly
# from Phase 2 while reusing the agent implementations.
phase_1_path = Path(__file__).resolve().parent.parent / "phase_1"
sys.path.insert(0, str(phase_1_path))

# TODO 1: Import all required agents
from workflow_agents.base_agents import (
    ActionPlanningAgent,
    KnowledgeAugmentedPromptAgent,
    EvaluationAgent,
    RoutingAgent,
)


# TODO 2: Load environment variables and OpenAI API key
load_dotenv(dotenv_path=phase_1_path / ".env")

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY is missing from the .env file.")


# TODO 3: Load the product specification
spec_path = (
    Path(__file__).resolve().parent
    / "Product-Spec-Email-Router.txt"
)

with open(spec_path, "r", encoding="utf-8") as file:
    product_spec = file.read()


# --------------------------------------------------
# Instantiate all agents
# --------------------------------------------------

required_feature_format = """
Return product features ONLY in this exact repeated structure:
Feature Name: <clear capability name>
Description: <what the feature does and why it exists>
Key Functionality: <specific capabilities or actions>
User Benefit: <value created for the user>
"""

required_task_format = """
Return engineering tasks ONLY in this exact repeated structure:
Task ID: <unique identifier>
Task Title: <brief development work title>
Related User Story: <matching user story>
Description: <technical work required>
Acceptance Criteria: <testable completion requirements>
Estimated Effort: <time or complexity estimate>
Dependencies: <required prior work or None>
"""

required_feature_labels = (
    "Feature Name",
    "Description",
    "Key Functionality",
    "User Benefit",
)

required_task_labels = (
    "Task ID",
    "Task Title",
    "Related User Story",
    "Description",
    "Acceptance Criteria",
    "Estimated Effort",
    "Dependencies",
)


def validate_repeated_structure(
    content,
    item_label,
    required_labels,
    section_name,
):
    """Raise an error unless every item uses the exact required labels."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    item_prefix = f"{item_label}:"

    if not lines or not lines[0].startswith(item_prefix):
        raise ValueError(
            f"{section_name} must contain only repeated structured items "
            f"starting with '{item_prefix}'."
        )

    item_starts = [
        index
        for index, line in enumerate(lines)
        if line.startswith(item_prefix)
    ]
    item_starts.append(len(lines))

    for item_number, start in enumerate(item_starts[:-1], start=1):
        item_lines = lines[start:item_starts[item_number]]
        found_labels = []

        for line in item_lines:
            if ":" not in line:
                continue

            label, value = line.split(":", 1)
            if label in required_labels or label.replace(" ", "").isalpha():
                found_labels.append(label)
                if not value.strip():
                    raise ValueError(
                        f"{section_name} item {item_number} has an empty "
                        f"'{label}:' value."
                    )

        if tuple(found_labels) != tuple(required_labels):
            raise ValueError(
                f"{section_name} item {item_number} labels must be exactly "
                f"{', '.join(label + ':' for label in required_labels)} "
                f"in order; found {', '.join(label + ':' for label in found_labels)}."
            )

# Action Planning Agent

knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)

# TODO 4: Instantiate ActionPlanningAgent
action_planning_agent = ActionPlanningAgent(
    openai_api_key=openai_api_key,
    knowledge=knowledge_action_planning
)


# --------------------------------------------------
# Product Manager - Knowledge Agent
# --------------------------------------------------

persona_product_manager = (
    "You are a Product Manager, you are responsible "
    "for defining the user stories for a product."
)

# TODO 5: Append the product specification
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, "
    "an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, "
    "where the personas are the different users of the product. "
    f"\n\nProduct Specification:\n{product_spec}"
)

# TODO 6: Instantiate Product Manager Knowledge Agent
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager,
    knowledge=knowledge_product_manager
)


# --------------------------------------------------
# Product Manager - Evaluation Agent
# --------------------------------------------------

# TODO 7: Define persona, criteria, and instantiate
persona_product_manager_eval = (
    "You are an evaluation agent that checks "
    "the answers of other worker agents."
)

evaluation_criteria_product_manager = (
    "The answer should be stories that follow the following structure: "
    "As a [type of user], I want [an action or feature] "
    "so that [benefit/value]."
)

product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_product_manager_eval,
    evaluation_criteria=evaluation_criteria_product_manager,
    agent_to_evaluate=product_manager_knowledge_agent,
    max_interactions=10
)


# --------------------------------------------------
# Program Manager - Knowledge Agent
# --------------------------------------------------

persona_program_manager = (
    "You are a Program Manager, you are responsible "
    "for defining the features for a product."
)

knowledge_program_manager = (
    "Features of a product are defined by organizing "
    "similar user stories into cohesive groups. Each feature must "
    "describe a distinct capability, its purpose, concrete functionality, "
    "and user value. Return every feature using all required labels in "
    "the exact order shown; do not use numbered feature labels or add "
    "user-story-only summaries.\n"
    f"{required_feature_format}"
)

program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager,
    knowledge=knowledge_program_manager
)


# --------------------------------------------------
# Program Manager - Evaluation Agent
# --------------------------------------------------

persona_program_manager_eval = (
    "You are an evaluation agent that checks "
    "the answers of other worker agents."
)

# TODO 8: Instantiate Program Manager Evaluation Agent
evaluation_criteria_program_manager = (
    "Accept only a complete list of distinct product features. Every feature "
    "must include all four labels, in this exact order, with meaningful "
    "feature-specific content: Feature Name, Description, Key Functionality, "
    "User Benefit. Reject numbered/abbreviated feature summaries, missing or "
    "reordered fields, and any content outside the repeated feature structures.\n"
    f"Required format:\n{required_feature_format}"
)

program_manager_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_program_manager_eval,
    evaluation_criteria=evaluation_criteria_program_manager,
    agent_to_evaluate=program_manager_knowledge_agent,
    max_interactions=10
)


# --------------------------------------------------
# Development Engineer - Knowledge Agent
# --------------------------------------------------

persona_dev_engineer = (
    "You are a Development Engineer, you are responsible "
    "for defining the development tasks for a product."
)

knowledge_dev_engineer = (
    "Development tasks are defined by identifying what "
    "needs to be built to implement each user story. Give each task a unique "
    "ID, link it to the matching user story, make acceptance criteria "
    "testable, estimate effort, and state dependencies explicitly. Return "
    "every task using all required labels in the exact order shown.\n"
    f"{required_task_format}"
)

development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer,
    knowledge=knowledge_dev_engineer
)


# --------------------------------------------------
# Development Engineer - Evaluation Agent
# --------------------------------------------------

persona_dev_engineer_eval = (
    "You are an evaluation agent that checks "
    "the answers of other worker agents."
)

# TODO 9: Instantiate Development Engineer Evaluation Agent
evaluation_criteria_dev_engineer = (
    "Accept only engineering tasks with all seven labels in this exact order "
    "for every task: Task ID, Task Title, Related User Story, Description, "
    "Acceptance Criteria, Estimated Effort, Dependencies. IDs must be unique, "
    "related stories must match the supplied user stories, criteria must be "
    "testable, and dependencies must name prior work or say None. Reject "
    "missing/reordered fields, numbered task summaries, and content outside "
    "the repeated task structures.\n"
    f"Required format:\n{required_task_format}"
)

development_engineer_evaluation_agent = EvaluationAgent(
    openai_api_key=openai_api_key,
    persona=persona_dev_engineer_eval,
    evaluation_criteria=evaluation_criteria_dev_engineer,
    agent_to_evaluate=development_engineer_knowledge_agent,
    max_interactions=10
)


# --------------------------------------------------
# Shared workflow state
# --------------------------------------------------

generated_user_stories = ""
generated_features = ""


# --------------------------------------------------
# TODO 11: Define agent support functions
# --------------------------------------------------

def product_manager_support_function(query):
    global generated_user_stories

    print("\n[Product Manager] Generating user stories")

    # Step 1: Generate initial response
    initial_response = product_manager_knowledge_agent.respond(query)

    # Step 2: Evaluate and refine the worker response
    # EvaluationAgent calls its worker again during evaluation.
    evaluation_result = product_manager_evaluation_agent.evaluate(query)

    # Step 3: Extract final evaluated response
    final_response = (
        evaluation_result.get("final_response")
        or initial_response
    )

    generated_user_stories = final_response

    # Step 4: Return the result with its originating section
    return {
        "role": "User Stories",
        "content": final_response,
        "evaluation": evaluation_result.get("evaluation")
    }


def program_manager_support_function(query):
    global generated_features

    print("\n[Program Manager] Generating product features")

    # Provide product specification and previously generated stories
    program_manager_knowledge_agent.knowledge = (
        f"{knowledge_program_manager}\n\n"
        f"Product Specification:\n{product_spec}\n\n"
        f"User Stories:\n{generated_user_stories}\n\n"
        "Group the user stories into cohesive product features "
        "using the required feature structure. Final formatting instruction: "
        "return features only. Use these exact labels, including the colon, "
        "for every item, in this order: Feature Name:, Description:, "
        "Key Functionality:, User Benefit:. "
        "Repeat the full structure for each feature; include no alternate "
        "feature or user-story summary formats.\n"
        f"{required_feature_format}"
    )

    # Step 1: Generate response
    initial_response = program_manager_knowledge_agent.respond(query)

    # Step 2: Evaluate response
    evaluation_result = program_manager_evaluation_agent.evaluate(query)

    # Step 3: Extract final response
    final_response = (
        evaluation_result.get("final_response")
        or initial_response
    )

    validate_repeated_structure(
        final_response,
        "Feature Name",
        required_feature_labels,
        "Product Features",
    )

    generated_features = final_response

    # Step 4: Return result
    return {
        "role": "Product Features",
        "content": final_response,
        "evaluation": evaluation_result.get("evaluation")
    }


def development_engineer_support_function(query):

    print("\n[Development Engineer] Generating engineering tasks")

    # Provide earlier deliverables for traceability
    development_engineer_knowledge_agent.knowledge = (
        f"{knowledge_dev_engineer}\n\n"
        f"Product Specification:\n{product_spec}\n\n"
        f"User Stories:\n{generated_user_stories}\n\n"
        f"Product Features:\n{generated_features}\n\n"
        "Define implementation tasks with references to "
        "the corresponding user stories. Final formatting instruction: return "
        "engineering tasks only. Use these exact labels, including the colon, "
        "for every item, in this order: Task ID:, Task Title:, "
        "Related User Story:, Description:, Acceptance Criteria:, "
        "Estimated Effort:, Dependencies:. Repeat the full "
        "structure for each task and include no alternate task summary format.\n"
        f"{required_task_format}"
    )

    # Step 1: Generate response
    initial_response = development_engineer_knowledge_agent.respond(query)

    # Step 2: Evaluate response
    evaluation_result = development_engineer_evaluation_agent.evaluate(query)

    # Step 3: Extract final response
    final_response = (
        evaluation_result.get("final_response")
        or initial_response
    )

    validate_repeated_structure(
        final_response,
        "Task ID",
        required_task_labels,
        "Engineering Tasks",
    )

    # Step 4: Return result
    return {
        "role": "Engineering Tasks",
        "content": final_response,
        "evaluation": evaluation_result.get("evaluation")
    }


# --------------------------------------------------
# TODO 10: Configure RoutingAgent
# --------------------------------------------------

agents = [
    {
        "name": "Product Manager",
        "description": (
            "Define product requirements and write user stories "
            "using personas, actions, and desired outcomes."
        ),
        "func": product_manager_support_function
    },
    {
        "name": "Program Manager",
        "description": (
            "Define product features by organizing related "
            "user stories into functional groups."
        ),
        "func": program_manager_support_function
    },
    {
        "name": "Development Engineer",
        "description": (
            "Define engineering tasks, implementation work, "
            "technical dependencies, acceptance criteria, "
            "and estimated development effort."
        ),
        "func": development_engineer_support_function
    }
]

routing_agent = RoutingAgent(
    openai_api_key=openai_api_key,
    agents=agents
)

routing_agent.agents = agents


# --------------------------------------------------
# Run the workflow
# --------------------------------------------------

print("\n*** Workflow execution started ***\n")

workflow_prompt = (
    "Produce a complete project plan for the Email Router product, "
    "including user stories, product features, and engineering tasks."
)

print(
    f"Task to complete in this workflow, "
    f"workflow prompt = {workflow_prompt}"
)


# --------------------------------------------------
# TODO 12: Execute workflow steps
# --------------------------------------------------

print("\nDefining workflow steps from the workflow prompt")

workflow_steps = action_planning_agent.extract_steps_from_prompt(
    workflow_prompt
)

# Remove any empty steps
workflow_steps = [
    step.strip()
    for step in workflow_steps
    if step.strip()
]

print("\nExtracted Workflow Steps:")

for i, step in enumerate(workflow_steps, start=1):
    print(f"{i}. {step}")


# Initialize the completed steps collection
completed_steps = []

# Execute each extracted step
for i, step in enumerate(workflow_steps, start=1):

    print(f"\n{'=' * 60}")
    print(f"Executing Workflow Step {i}")
    print(f"{'=' * 60}")
    print(f"Step: {step}")

    # Route only the short step text.
    # Do not pass accumulated results to the router.
    result = routing_agent.route(step)

    # Store results with role metadata
    if isinstance(result, dict):
        completed_steps.append(result)

        print(f"\nCompleted By: {result['role']}")
        print(f"\nResult:\n{result['content']}")
    else:
        print(f"\nRouting Result:\n{result}")


# --------------------------------------------------
# TODO 13: Assemble consolidated project plan
# --------------------------------------------------

print("\n*** Consolidated Project Plan ***\n")

sections = (
    "User Stories",
    "Product Features",
    "Engineering Tasks"
)

consolidated_plan = []

consolidated_plan.append(
    "EMAIL ROUTER - CONSOLIDATED PROJECT PLAN"
)

consolidated_plan.append("=" * 60)

for section in sections:

    print(f"\n--- {section} ---\n")

    consolidated_plan.append(f"\n--- {section} ---\n")

    section_results = [
        entry["content"]
        for entry in completed_steps
        if entry.get("role") == section
    ]

    if section_results:
        for result in section_results:
            print(result)
            print()
            consolidated_plan.append(result)
    else:
        missing_message = (
            f"No completed results available for {section}."
        )
        print(missing_message)
        consolidated_plan.append(missing_message)


# Validate structured sections after the consolidated plan is assembled.
for section, item_label, required_labels in (
    (
        "Product Features",
        "Feature Name",
        required_feature_labels,
    ),
    (
        "Engineering Tasks",
        "Task ID",
        required_task_labels,
    ),
):
    section_content = "\n\n".join(
        entry["content"]
        for entry in completed_steps
        if entry.get("role") == section
    )
    validate_repeated_structure(
        section_content,
        item_label,
        required_labels,
        section,
    )


# Optionally save the consolidated plan
output_path = (
    Path(__file__).resolve().parent
    / "Email-Router-Consolidated-Project-Plan.txt"
)

with open(output_path, "w", encoding="utf-8") as file:
    file.write("\n\n".join(consolidated_plan))

print(f"\nConsolidated project plan saved to: {output_path}")
print("\n*** Workflow execution completed ***")
