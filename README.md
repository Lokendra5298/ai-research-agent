# AI Research Paper Writing Agent

A conversational research assistant built with **Google Gemini**, **LangChain/LangGraph**, and **Streamlit**. It searches arXiv, reads research-paper PDFs, discusses research gaps, suggests research directions, and saves a manuscript as a **LaTeX `.tex` file** when requested.

> **Current output:** LaTeX source only. PDF compilation is **not** required, and no `sudo`, `latexmk`, or TeX installation is needed.

## Features

- Interactive, multi-turn Streamlit chat interface.
- arXiv paper search with titles, authors, abstracts, and PDF URLs.
- Text extraction from research-paper PDF links.
- Research paper analysis, gap identification, and idea generation.
- Generation and saving of LaTeX manuscripts to `output/`.
- Conversation context through LangGraph's `InMemorySaver` (while the Python process is running).

## Project structure

```text
p7_write_research_paper/
├── main.py                 # Gemini model, tools, prompt, agent graph
├── app.py                  # Streamlit chat interface
├── arxiv_tool.py           # arxiv_search tool
├── read_pdf.py             # read_pdf tool
├── write_pdf.py            # render_latex_pdf tool (saves .tex only)
├── requirements.txt
├── .env                    # Local secrets; do not commit
├── .gitignore              # Recommended
└── output/                 # Generated LaTeX files
```

> The name `render_latex_pdf` is retained for compatibility with existing agent code, but the function should **write `.tex` only**, not compile a PDF.

## Prerequisites

- Python **3.12** (other supported Python 3 versions may work with compatible dependencies).
- Internet access to Google Gemini and arXiv APIs.
- A Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey).
- No administrator privileges required when using a Python virtual environment.

## Installation

From the project directory:

```bash
cd ~/Learning_Codes/GenAI_Projects/p7_write_research_paper
```

Activate the existing virtual environment if you already have one:

```bash
source ~/Learning_Codes/lvenv/bin/activate
```

Otherwise, create one within the project:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Verify the key imports:

```bash
python -c "from langchain_google_genai import ChatGoogleGenerativeAI; from langchain.agents import create_agent; import streamlit; print('Imports OK')"
```

## Configure the Gemini API key

Create `.env` in the project root:

```env
GOOGLE_API_KEY=replace_with_your_gemini_api_key
GOOGLE_MODEL=replace_with_a_model_available_to_your_project
```

Check available models and quotas in [Google AI Studio](https://aistudio.google.com/) before setting `GOOGLE_MODEL`. The code's default model must also match one your project can access. The previous example used `gemini-3.5-flash-lite`; model access, naming, and free-tier quotas can change.

On a shared server, protect the secrets file:

```bash
chmod 600 .env
```

Do **not** put your real API key in source code or version control.

Your `main.py` should load the environment:

```python
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv(override=True)

model = ChatGoogleGenerativeAI(
    model=os.environ["GOOGLE_MODEL"],
    api_key=os.environ["GOOGLE_API_KEY"],
    max_retries=2,
)
```

Do not set `temperature` for Gemini models that report fixed sampling defaults.

## Run the terminal chat

```bash
python main.py
```

The interactive terminal chat should be guarded by:

```python
if __name__ == "__main__":
    main()
```

This is important because `app.py` imports `graph` from `main.py`; importing it must **not** start the terminal input loop.

## Run the Streamlit chat app

On the server:

```bash
python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Streamlit then listens **on the server**, not automatically on your laptop.

### Access a remote server with SSH forwarding

On your **local computer**, in a separate terminal:

```bash
ssh -N -L 8501:127.0.0.1:8501 lk57288c@YOUR_SERVER_HOST
```

Replace `YOUR_SERVER_HOST` with the server's actual SSH hostname or IP address. Keep the tunnel and Streamlit processes running, then open:

**http://localhost:8501**

If port 8501 is busy locally, use a different *local* port, for example:

```bash
ssh -N -L 8502:127.0.0.1:8501 lk57288c@YOUR_SERVER_HOST
```

Then visit **http://localhost:8502**.

### Run Streamlit after disconnecting from SSH (optional)

```bash
nohup python -m streamlit run app.py \
  --server.address 127.0.0.1 --server.port 8501 \
  > streamlit.log 2>&1 &
```

Inspect logs:

```bash
tail -n 50 streamlit.log
```

## Example conversation

1. `Find recent arXiv papers on transformer-based medical image segmentation.`
2. `Summarize the methods and PDF links for the five papers.`
3. `Read the second paper and explain its methodology and limitations.`
4. `Suggest three feasible research directions based on its limitations.`
5. `Develop a proposed methodology for idea 1.`
6. `Write a LaTeX research manuscript for idea 1 and save the .tex file.`

The agent calls tools automatically when needed. The `app.py` interface is primarily for **chat**, not a download portal. Generated `.tex` files are saved on the server in `output/`.

## Tools

| Tool | Module | Purpose |
|---|---|---|
| `arxiv_search` | `arxiv_tool.py` | Retrieve paper metadata from the arXiv API |
| `read_pdf` | `read_pdf.py` | Download and extract text from a PDF |
| `render_latex_pdf` | `write_pdf.py` | Save complete LaTeX text to a `.tex` file |

**Important:** The current `arxiv_search` tool should return a `list[dict]`; `read_pdf` should return a `str`; and `render_latex_pdf` should return the saved file path as a `str`. Functions must be registered as LangChain tools (using `@tool` or `tool(...)`) before adding them to the agent.

## Conversation memory

The LangGraph agent uses an in-memory checkpointer and a unique `thread_id` per Streamlit session. `app.py` should send **only the latest user message** on each turn; the checkpointer supplies prior messages.

`InMemorySaver` does **not** persist conversations through server restarts and is not a persistent, cross-worker storage system.

## Troubleshooting

| Problem | Suggested fix |
|---|---|
| `ModuleNotFoundError: langchain_google_genai` | `python -m pip install -r requirements.txt` in the active environment |
| `404 NOT_FOUND` for Gemini model | Change `GOOGLE_MODEL` to an API-accessible model |
| `429 RESOURCE_EXHAUSTED`, quota `limit: 0` | Check project/model quota or billing; waiting may not help when the limit is zero |
| Gemini warning that `temperature` is ignored | Remove the `temperature` argument |
| arXiv query rejects quotation marks | Normalize queries in `arxiv_tool.py` rather than raising on quotes |
| Agent searches arXiv repeatedly | Add tool-use instructions and/or `ToolCallLimitMiddleware` |
| Streamlit page keeps loading | Check `ps aux | grep '[s]treamlit'`, server logs, and SSH forwarding |
| `input()` blocks the web app | Wrap terminal `main()` invocation in `if __name__ == '__main__':` |
| `_get_client_ip: command not found` | Separate server shell configuration warning; not a Gemini/Streamlit error |

Quick Streamlit health checks on the server:

```bash
ps aux | grep '[s]treamlit'
ss -ltn | grep 8501
curl -I http://127.0.0.1:8501
```

## Research integrity and limitations

- Use actual arXiv papers and verify references and PDF links.
- Never invent authors, citations, measurements, experimental data, or results.
- Clearly distinguish **proposed experiments** from completed experiments.
- arXiv manuscripts are often preprints and may not be peer-reviewed.
- `PyPDF2` cannot reliably extract scanned pages, complex equations, tables, or figures.
- A limited arXiv search cannot establish that a proposed idea is globally novel.
- If handling private manuscripts, consider the privacy implications of sending text to external APIs.
- Untrusted LaTeX should not be compiled on the server without isolation.

## Security

Recommended `.gitignore`:

```gitignore
.env
.venv/
lvenv/
__pycache__/
*.pyc
streamlit.log
output/
```

If multiple users share the server or the Streamlit app, isolate generated files per user/session and apply appropriate access controls before exposing downloads.

## References

- [Gemini API documentation](https://ai.google.dev/gemini-api/docs)
- [Google AI Studio](https://aistudio.google.com/)
- [LangChain documentation](https://docs.langchain.com/)
- [LangGraph documentation](https://docs.langchain.com/oss/python/langgraph/overview)
- [Streamlit documentation](https://docs.streamlit.io/)
- [arXiv API documentation](https://info.arxiv.org/help/api/index.html)
