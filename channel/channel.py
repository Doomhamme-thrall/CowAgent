"""
Message sending channel abstract class
"""

from bridge.bridge import Bridge
from bridge.context import Context
from bridge.reply import *
from common.log import logger
from config import conf


class Channel(object):
    channel_type = ""
    NOT_SUPPORT_REPLYTYPE = [ReplyType.VOICE, ReplyType.IMAGE]

    def __init__(self):
        import threading
        self._startup_event = threading.Event()
        self._startup_error = None
        self.cloud_mode = False  # set to True by ChannelManager when running with cloud client

    def startup(self):
        """
        init channel
        """
        raise NotImplementedError

    def report_startup_success(self):
        self._startup_error = None
        self._startup_event.set()

    def report_startup_error(self, error: str):
        self._startup_error = error
        self._startup_event.set()

    def wait_startup(self, timeout: float = 3) -> (bool, str):
        """
        Wait for channel startup result.
        Returns (success: bool, error_msg: str).
        """
        ready = self._startup_event.wait(timeout=timeout)
        if not ready:
            return True, ""
        if self._startup_error:
            return False, self._startup_error
        return True, ""

    def stop(self):
        """
        stop channel gracefully, called before restart
        """
        pass

    def handle_text(self, msg):
        """
        process received msg
        :param msg: message object
        """
        raise NotImplementedError

    # 统一的发送函数，每个Channel自行实现，根据reply的type字段发送不同类型的消息
    def send(self, reply: Reply, context: Context):
        """
        send message to user
        :param msg: message content
        :param receiver: receiver channel account
        :return:
        """
        raise NotImplementedError

    def build_reply_content(self, query, context: Context = None) -> Reply:
        """
        Build reply content, using agent if enabled in config
        """
        # Check if agent mode is enabled
        use_agent = conf().get("agent", False)

        if use_agent:
            try:
                logger.info("[Channel] Using agent mode")

                # Add channel_type to context if not present
                if context and "channel_type" not in context:
                    context["channel_type"] = self.channel_type

                # Ensure agent/session routing metadata exists even when a
                # channel overrides _compose_context and skips ChatChannel logic.
                if context:
                    default_agent_id = str(conf().get("default_agent_id", "main") or "main").strip() or "main"
                    resolved_agent_id = str(context.get("agent_id", "") or "").strip()
                    if not resolved_agent_id:
                        resolver = getattr(self, "_resolve_agent_id", None)
                        if callable(resolver):
                            try:
                                resolved_agent_id = str(resolver(context) or "").strip()
                            except Exception:
                                resolved_agent_id = ""
                    if not resolved_agent_id:
                        bindings = conf().get("channel_agent_bindings", {}) or {}
                        ch_type = str(context.get("channel_type", "") or self.channel_type or "").strip()
                        if isinstance(bindings, dict):
                            resolved_agent_id = str(bindings.get(ch_type, "") or "").strip()
                    if not resolved_agent_id:
                        resolved_agent_id = default_agent_id

                    session_key = context.get("session_key")
                    origin_session_id = context.get("origin_session_id")
                    legacy_session_id = context.get("session_id")
                    if not origin_session_id and legacy_session_id:
                        legacy_session_id = str(legacy_session_id)
                        prefix = f"{resolved_agent_id}:"
                        origin_session_id = legacy_session_id[len(prefix):] if legacy_session_id.startswith(prefix) else legacy_session_id
                    if not session_key and origin_session_id:
                        session_key = f"{resolved_agent_id}:{origin_session_id}"

                    context["agent_id"] = resolved_agent_id
                    if origin_session_id:
                        context["origin_session_id"] = str(origin_session_id)
                    if session_key:
                        context["session_key"] = str(session_key)
                        context["session_id"] = str(session_key)

                    # Keep kwargs synchronized because AgentBridge reads both.
                    try:
                        if hasattr(context, "kwargs") and isinstance(context.kwargs, dict):
                            context.kwargs["agent_id"] = resolved_agent_id
                            if origin_session_id:
                                context.kwargs["origin_session_id"] = str(origin_session_id)
                            if session_key:
                                context.kwargs["session_key"] = str(session_key)
                                context.kwargs["session_id"] = str(session_key)
                    except Exception:
                        pass

                # Read on_event callback injected by the channel (e.g. web SSE)
                on_event = context.get("on_event") if context else None

                # Use agent bridge to handle the query
                return Bridge().fetch_agent_reply(
                    query=query,
                    context=context,
                    on_event=on_event,
                    clear_history=False
                )
            except Exception as e:
                logger.error(f"[Channel] Agent mode failed, fallback to normal mode: {e}")
                # Fallback to normal mode if agent fails
                return Bridge().fetch_reply_content(query, context)
        else:
            # Normal mode
            return Bridge().fetch_reply_content(query, context)

    def build_voice_to_text(self, voice_file) -> Reply:
        return Bridge().fetch_voice_to_text(voice_file)

    def build_text_to_voice(self, text) -> Reply:
        return Bridge().fetch_text_to_voice(text)
