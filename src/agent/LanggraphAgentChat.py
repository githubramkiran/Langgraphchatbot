
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver,InMemorySaver
#from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import os
import getpass
from langchain.chat_models import init_chat_model
from langchain_google_genai import ChatGoogleGenerativeAI

import logging
import sys

# Configure logger to output info messages to stdout for Render
logging.basicConfig(format="%(message)s", stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger("LanggraphChatbot_APP")
logger.info("Info - Application initialized successfully")


if not os.environ.get("GOOGLE_API_KEY"):
  os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
#llm = init_chat_model("google_genai:gemini-2.5-flash-lite")
llm = ChatGoogleGenerativeAI(model="gemini-3.7-flash")
# 1. Define Graph & Memory (Same as above)
#llm = ChatOpenAI(model="gpt-4o")
#checkpointer = MemorySaver()  # Use PostgresSaver for production
checkpointer = InMemorySaver()

def call_model(state: MessagesState):
    return {"messages": llm.invoke(state["messages"])}


builder = StateGraph(MessagesState)
builder.add_node("chatbot", call_model)
builder.add_edge(START, "chatbot")
#graph = builder.compile()
graph = builder.compile(checkpointer=checkpointer)

# 2. Define Request Schema
class ChatRequest(BaseModel):
    message: str
    thread_id: str = "default_thread"


# 3. Create Endpoint
app = FastAPI()


@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        # Prepare input state
        input_state = {"messages": [HumanMessage(content=request.message)]}
        #print('request.message:',request.message)
        logger.info(f"request message:{request.message}")
        # Prepare config for state persistence
        config = {"configurable": {"thread_id": request.thread_id}}

        # Invoke Graph
        # Use ainvoke for non-blocking async execution
        result = await graph.ainvoke(input_state, config=config)

        # Extract the last message content
        last_message = result["messages"][-1].content
        logger.info(f"response:{last_message}")
        return {"response": last_message, "thread_id": request.thread_id}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    #uvicorn.run("LanggraphAgentChat:app", host="127.0.0.1", port=8000,reload=True)
    uvicorn.run(app, host="0.0.0.0", port=8000)
