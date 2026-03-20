"""
DevOps RAG Agent — Harshad Khetpal
A LangChain agent that answers questions about infrastructure by
querying a vector store of runbooks, incident reports, and K8s docs.
"""
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from typing import Optional
import subprocess
import json


# --- Tools ---

@tool
def query_runbook_knowledge_base(query: str) -> str:
    """Search the internal runbook and incident knowledge base for relevant information."""
    # In production: connect to your Chroma/Qdrant/Pinecone instance
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma(
        collection_name="runbooks",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )
    docs = vectorstore.similarity_search(query, k=3)
    return "\n\n".join([d.page_content for d in docs])


@tool
def get_kubernetes_pod_status(namespace: str = "default", label_selector: Optional[str] = None) -> str:
    """Get the status of Kubernetes pods in a namespace. Use for troubleshooting."""
    cmd = ["kubectl", "get", "pods", "-n", namespace, "-o", "json"]
    if label_selector:
        cmd.extend(["-l", label_selector])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        pods = json.loads(result.stdout)
        summary = []
        for pod in pods.get("items", []):
            name = pod["metadata"]["name"]
            phase = pod["status"].get("phase", "Unknown")
            summary.append(f"{name}: {phase}")
        return "\n".join(summary) if summary else "No pods found"
    except Exception as e:
        return f"Error querying cluster: {str(e)}"


@tool
def check_recent_alerts(service: str, hours: int = 1) -> str:
    """Check recent PagerDuty/Alertmanager alerts for a given service."""
    # Placeholder for real alert API integration
    return f"No critical alerts for {service} in the last {hours} hour(s)."


# --- Agent Setup ---

def build_devops_agent(model: str = "gpt-4o"):
    llm = ChatOpenAI(model=model, temperature=0)
    tools = [query_runbook_knowledge_base, get_kubernetes_pod_status, check_recent_alerts]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a senior DevOps/SRE assistant for Harshad Khetpal's infrastructure.
You have access to runbooks, Kubernetes cluster state, and alert systems.
Always check relevant documentation before suggesting fixes.
Be concise, precise, and production-safe in your recommendations."""),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ])

    agent = create_openai_tools_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=5)


if __name__ == "__main__":
    agent = build_devops_agent()
    response = agent.invoke({
        "input": "The payment service pods in the production namespace are restarting frequently. What should I check first?"
    })
    print(response["output"])
