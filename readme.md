### Prerequisites

You need `uv` installed on your system. `uv` is a fast Python package installer and resolver.

If you don't have `uv` installed, you can get it using one of the following methods:

*   **Recommended (standalone `uv`):**
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
    This script will install `uv` into `~/.cargo/bin` and print instructions to add this directory to your system's `PATH`. Follow those instructions for `uv` to be available globally.

*   **Using `pip` (if you already have Python and pip):**
    ```bash
    pip install uv
    ```
### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/jchavezar/agentengine-langchain.git
    cd agent_engine
    ```

2.  **Install project dependencies:**
    `uv` will automatically create a virtual environment (typically in `./.venv` in your project root) and install all required packages based on your `uv.lock` file and `pyproject.toml`.

    ```bash
    uv sync
    ```

```bash
source /Users/jesusarguelles/IdeaProjects/hackathon-adk-on-gcp/a2a/.venv/bin/activate
uv pip install flet "flet[all]==0.28.3"
```

### Configure and Test Langgraph Locally

[agent.py](agent.py): has a local testing code, the only requirements is to have either service account authenticated
or use ADC (Application Default Credentials) in your local PC.

Something like this is returned:

![img.png](screenshot_1.png)

### Deploy the Tested Model in Agent Engine

_We are adding open telemetry package to traceability._

[agent_engine.py](agent_engine.py), takes the agent tested before and wrap it in a Custom Class so it can be deployed in
Agent Engine.

Once deployed or updated, the output should look like this:

![img_1.png](screenshot_2.png)

Next step is to plug the Agent Engine into a Custom UI:

```bash
flet run frontend_agent_engine.py
```

[frontend_agent_engine.py](frontend_agent_engine.py)

![img_2.png](screenshot_3.png)