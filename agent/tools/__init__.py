"""Tools module for Agent.

This package exposes the built-in tools used by the agent runtime while
keeping optional dependencies isolated so a missing extra does not break
the whole package import.
"""

from common.log import logger

from agent.tools.base_tool import BaseTool
from agent.tools.tool_manager import ToolManager
from agent.tools.read.read import Read
from agent.tools.write.write import Write
from agent.tools.edit.edit import Edit
from agent.tools.bash.bash import Bash
from agent.tools.ls.ls import Ls
from agent.tools.send.send import Send
from agent.tools.memory.memory_search import MemorySearchTool
from agent.tools.memory.memory_get import MemoryGetTool


def _safe_import(name: str, import_fn, missing_message: str, log_level: str = "error"):
    try:
        return import_fn()
    except ImportError as e:
        message = f"[Tools] {name} not loaded - missing dependency: {e}\n{missing_message}"
        if log_level == "info":
            logger.info(message)
        elif log_level == "warning":
            logger.warning(message)
        else:
            logger.error(message)
    except Exception as e:
        logger.error(f"[Tools] {name} failed to load: {e}", exc_info=True)
    return None


EnvConfig = _safe_import(
    "EnvConfig",
    lambda: __import__("agent.tools.env_config.env_config", fromlist=["EnvConfig"]).EnvConfig,
    "  To enable environment variable management, run:\n    pip install python-dotenv>=1.0.0",
)

SchedulerTool = _safe_import(
    "SchedulerTool",
    lambda: __import__("agent.tools.scheduler.scheduler_tool", fromlist=["SchedulerTool"]).SchedulerTool,
    "  To enable scheduled tasks, run:\n    pip install croniter>=2.0.0",
)

WebSearch = _safe_import(
    "WebSearch",
    lambda: __import__("agent.tools.web_search.web_search", fromlist=["WebSearch"]).WebSearch,
    "",
)

WebFetch = _safe_import(
    "WebFetch",
    lambda: __import__("agent.tools.web_fetch.web_fetch", fromlist=["WebFetch"]).WebFetch,
    "",
)

Vision = _safe_import(
    "Vision",
    lambda: __import__("agent.tools.vision.vision", fromlist=["Vision"]).Vision,
    "",
)

ImageGenerate = _safe_import(
    "ImageGenerate",
    lambda: __import__("agent.tools.image_generate.image_generate", fromlist=["ImageGenerate"]).ImageGenerate,
    "",
)

GoogleSearch = None
FileSave = None
Terminal = None

__all__ = [
    "BaseTool",
    "ToolManager",
    "Read",
    "Write",
    "Edit",
    "Bash",
    "Ls",
    "Send",
    "MemorySearchTool",
    "MemoryGetTool",
    "EnvConfig",
    "SchedulerTool",
    "WebSearch",
    "WebFetch",
    "Vision",
    "ImageGenerate",
    "GoogleSearch",
    "FileSave",
    "Terminal",
]
