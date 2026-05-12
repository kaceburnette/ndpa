"""
NDPA — predictive memory layer for AI.

  from ndpa import Client
  client = Client(api_key="ndpa_...")
  client.log_turn(session_id="chat_1", role="user", content="Hello")
"""

from .client import Client, NDPAError

__version__ = "0.2.0"
__all__ = ["Client", "NDPAError"]
