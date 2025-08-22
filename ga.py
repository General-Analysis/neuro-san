import os
import sys
from pathlib import Path
from typing import Dict, Any

# Set up paths BEFORE any neuro_san imports
script_dir = Path(__file__).parent
os.environ['AGENT_TOOL_PATH'] = str(script_dir / "neuro_san" / "coded_tools")
# Also set PYTHONPATH for neuro-san's tool resolution
os.environ['PYTHONPATH'] = str(script_dir) + os.pathsep + os.environ.get('PYTHONPATH', '')
# Add parent directory to Python path so modules can be imported
sys.path.insert(0, str(script_dir))

# Now load environment variables
from dotenv import load_dotenv
load_dotenv()

# NOW import neuro_san modules after environment is set up
from neuro_san.client.agent_session_factory import AgentSessionFactory
from neuro_san.client.streaming_input_processor import StreamingInputProcessor
from neuro_san.interfaces.agent_session import AgentSession
from neuro_san.message_processing.basic_message_processor import BasicMessageProcessor


class AgentClient:
    """
    Client for interacting with NeuroSAN agents directly.
    """

    def __init__(self, agent: str, thinking_dir: str = None):
        """
        Constructor

        :param agent: The name of the agent to talk to
        :param thinking_dir: Directory to write agent thinking logs
        """
        self.agent: str = agent
        self.thinking_dir: str = thinking_dir

    def get_answer_for(self, text: str) -> str:
        """
        Sends text to the agent and returns the agent's answer
        :param text: The text to send to the agent
        :return: The text answer from the agent.
        """
        session: AgentSession = AgentSessionFactory().create_session("direct", self.agent)
        input_processor = StreamingInputProcessor(session=session, thinking_dir=self.thinking_dir)
        processor: BasicMessageProcessor = input_processor.get_message_processor()
        # Add MAXIMAL filter to get all messages including tool calls
        request: Dict[str, Any] = input_processor.formulate_chat_request(text)
        request["chat_filter"] = {"chat_filter_type": "MAXIMAL"}

        # Call streaming_chat()
        for chat_response in session.streaming_chat(request):
            message: Dict[str, Any] = chat_response.get("response", {})
            if message:
                processor.process_message(message, message.get("type"))

        raw_answer: str = processor.get_compiled_answer()

        return raw_answer

if __name__ == "__main__":
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    thinking_dir = script_dir / "thinking"
    
    # Create thinking directory if it doesn't exist
    thinking_dir.mkdir(exist_ok=True)
    
    client = AgentClient(
        agent="intranet_agents_with_tools",
        thinking_dir=str(thinking_dir),
    )
    query = """
    I'd like to check my leave balance
    """
    ans = client.get_answer_for(query)
    print(ans)