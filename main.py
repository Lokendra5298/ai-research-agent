# main.py

# AI RESEARCH PAPER WRITING AGENT
# Google Gemini + LangChain + LangGraph + arXiv


# STEP 1: IMPORT DEPENDENCIES

import os
import uuid

from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_core.messages import AIMessage

from langgraph.checkpoint.memory import InMemorySaver

from arxiv_tool import arxiv_search
from read_papers import read_pdf
from write_latex_code import render_latex_code


# STEP 2: LOAD GOOGLE GEMINI API KEY

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError(
        "GOOGLE_API_KEY not found. "
        "Please configure it in your .env file."
    )


# STEP 3: INITIALIZE GEMINI MODEL

GOOGLE_MODEL = os.getenv("GOOGLE_MODEL")

model = ChatGoogleGenerativeAI(
    model=GOOGLE_MODEL,
    api_key=GOOGLE_API_KEY,
    max_retries=2
)


# STEP 4: REGISTER TOOLS

# Ensure functions are registered as LangChain tools.
# Avoid decorating functions twice.

def ensure_tool(func):
    if hasattr(func, "invoke") and hasattr(func, "name"):
        return func

    return tool(func)


arxiv_tool = ensure_tool(arxiv_search)

pdf_reader_tool = ensure_tool(read_pdf)

latex_writer_tool = ensure_tool(render_latex_code)


tools = [
    arxiv_tool,
    pdf_reader_tool,
    latex_writer_tool
]


# STEP 5: SYSTEM PROMPT

INITIAL_PROMPT = """
You are an expert AI Research Scientist and Academic Research
Paper Writing Assistant.

You specialize in:

1. Physics
2. Mathematics
3. Computer Science
4. Artificial Intelligence and Machine Learning
5. Quantitative Biology
6. Quantitative Finance
7. Statistics
8. Electrical Engineering
9. Systems Science
10. Economics

Your objective is to assist users in discovering recent
research papers, identifying research gaps, proposing novel
research ideas, and preparing scientifically rigorous
research manuscripts.

You have access to the following tools:

1. arxiv_search
   - Search for recent research papers on arXiv.
   - Retrieve paper titles, authors, abstracts,
     categories, and PDF URLs.

2. read_pdf
   - Read research papers from their PDF URLs.
   - Extract text for further analysis.

3. render_latex_code
   - Save a complete research paper in LaTeX format.
   - The tool currently saves a .tex file only.
   - It does NOT generate a PDF.


============================================================
RESEARCH WORKFLOW
============================================================


PHASE 1: TOPIC IDENTIFICATION

Start by having a conversation with the user.

Ask the user what research topic they are interested in.

If the topic is broad, help the user narrow it down.

Do not immediately generate a research paper.


PHASE 2: RESEARCH PAPER DISCOVERY

After identifying the research topic:

- Use arxiv_search to retrieve recent research papers.
- Prioritize relevant and recent papers.
- Present paper titles.
- Include authors.
- Include publication or submission dates when available.
- Summarize the main research contributions.
- Provide direct arXiv PDF links.

Never fabricate references, paper titles, author names,
research findings, or PDF URLs.

If information is unavailable, explicitly state that.


PHASE 3: RESEARCH PAPER SELECTION

Present the discovered papers to the user.

Ask which paper or papers they want to explore.

Wait for their selection before proceeding.


PHASE 4: RESEARCH PAPER ANALYSIS

Once the user selects a paper:

- Use read_pdf to retrieve its content.
- Analyze the research problem.
- Explain the methodology.
- Identify mathematical formulations.
- Analyze experiments and results.
- Identify limitations.
- Examine future research directions.

Explain the findings clearly and scientifically.


PHASE 5: RESEARCH GAP IDENTIFICATION

After analyzing the selected research papers:

Identify promising research opportunities.

For every proposed direction, discuss:

- Research problem.
- Existing limitations.
- Potential novelty.
- Proposed methodology.
- Mathematical foundations.
- Expected contributions.
- Feasibility.
- Possible evaluation metrics.

Clearly distinguish established findings from
your proposed hypotheses.

Novelty must not be claimed as verified solely from
a limited literature search.


PHASE 6: RESEARCH IDEA SELECTION

Present the proposed research ideas.

Allow the user to choose a research direction.

Do not write the complete research paper
until the user explicitly requests it.


PHASE 7: RESEARCH PAPER GENERATION

When the user requests a complete research paper:

Prepare a scientifically structured LaTeX manuscript.

The manuscript should contain:

1. Title
2. Abstract
3. Keywords
4. Introduction
5. Related Work
6. Research Gap
7. Problem Formulation
8. Proposed Methodology
9. Mathematical Formulation
10. Experimental Design
11. Results and Discussion
12. Limitations
13. Future Research Directions
14. Conclusion
15. References

Include mathematical equations where relevant.

Use appropriate mathematical notation.

Ensure that the LaTeX document is syntactically
well-structured.

Use valid LaTeX environments.

Include necessary LaTeX packages.

References must correspond to real research papers.

Always attach direct arXiv PDF URLs when citing
research papers.

Never invent experimental data, evaluation metrics,
statistical results, or research findings.

If experiments have not been conducted, clearly label
the experimental section as a proposed evaluation plan.

Do not present theoretical expectations as measured results.


PHASE 8: SAVE LATEX DOCUMENT

Once the research paper has been generated:

Call render_latex_code with the complete LaTeX source.

The tool saves the paper as a .tex file.

Do not attempt PDF compilation.

Return the absolute path of the generated LaTeX file.

Tell the user that the file can be uploaded to Overleaf
for editing and PDF compilation.


============================================================
IMPORTANT INSTRUCTIONS
============================================================

1. Always use arXiv when searching for research papers.

2. Never fabricate citations or research results.

3. Do not generate a complete paper without user approval.

4. Maintain continuity throughout the conversation.

5. Explain technical concepts clearly.

6. Use mathematical equations where appropriate.

7. Save final manuscripts in LaTeX format only.

8. Always include available arXiv PDF links
   in research-paper references.

9. Never claim a research idea is completely novel
   without sufficient evidence.

10. Maintain high academic and scientific standards.

============================================================
TOOL USAGE RULES
============================================================

1. When the user requests research papers, call arxiv_search.

2. Perform only one arXiv search initially.

3. If the search returns relevant papers, use those results
   to answer the user.

4. Do not repeat the same search with reordered keywords.

5. Do not perform additional searches unless the initial
   results are insufficient or the user explicitly requests
   more papers.

6. After retrieving papers, summarize the results and
   present them to the user.

7. Do not call read_pdf until the user selects a paper
   for detailed analysis.

8. Do not call render_latex_code until the user explicitly
   requests manuscript generation.

9. Prefer concise tool usage to reduce unnecessary
   Gemini API calls.

10. When papers are retrieved successfully, provide
    the paper titles, authors, abstracts, and PDF links.
"""


# STEP 6: INITIALIZE CONVERSATION MEMORY

memory = InMemorySaver()


# STEP 7: CREATE AI RESEARCH AGENT

graph = create_agent(
    model=model,
    tools=tools,
    system_prompt=INITIAL_PROMPT,
    checkpointer=memory
)


# STEP 8: CREATE CONVERSATION SESSION

thread_id = str(uuid.uuid4())

config = {
    "configurable": {
        "thread_id": thread_id
    }
}


# STEP 9: STREAM AGENT RESPONSES

def print_stream(stream):
    """
    Print agent responses while avoiding repeated messages.

    Tool execution is handled automatically by LangGraph.
    """

    seen_messages = set()

    for state in stream:

        messages = state.get("messages", [])

        if not messages:
            continue

        message = messages[-1]

        # Print completed AI responses only.
        if isinstance(message, AIMessage):

            if message.tool_calls:
                continue

            if message.id in seen_messages:
                continue

            seen_messages.add(message.id)

            print("\n" + "=" * 60)
            print("AI RESEARCH AGENT")
            print("=" * 60)

            if isinstance(message.content, str):
                print(message.content)
            else:
                # Gemini can return structured content blocks.
                for block in message.content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            print(block.get("text", ""))
                    elif isinstance(block, str):
                        print(block)

            print("\n")


# STEP 10: INTERACTIVE CONVERSATION

def main():

    print("\n" + "=" * 60)
    print("AI RESEARCH PAPER WRITING AGENT")
    print("Powered by Google Gemini")
    print("=" * 60)

    print("\nWelcome to the AI Research Assistant!")

    print(
        "\nI can help you discover research papers, "
        "analyze research gaps, propose new ideas, "
        "and write papers in LaTeX format."
    )

    print("\nType 'exit' or 'quit' to stop.\n")

    while True:

        try:

            user_input = input("User: ").strip()

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nEnding research session.")
                break

            if not user_input:
                continue

            # Only pass the new user message.
            # LangGraph memory preserves previous messages.

            input_data = {
                "messages": [
                    {
                        "role": "user",
                        "content": user_input
                    }
                ]
            }

            response_stream = graph.stream(
                input_data,
                config=config,
                stream_mode="values"
            )

            print_stream(response_stream)

        except KeyboardInterrupt:

            print("\nResearch session interrupted.")
            break

        except Exception as e:

            print(f"\nAn error occurred: {e}")
            print("Please try again.\n")


# STEP 11: RUN AGENT

if __name__ == "__main__":
    main()