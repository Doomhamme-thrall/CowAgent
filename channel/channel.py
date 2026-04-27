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

                    # Apply channel/task default model profile when caller did
                    # not provide explicit model_override (e.g. QQ default model).
                    if not context.get("model_override"):
                        profile_key = ""
                        if bool(context.get("is_scheduled_task", False)):
                            profile_key = "default_model_task"
                        else:
                            ct = str(context.get("channel_type", "") or self.channel_type or "").strip().lower()
                            if ct == "web":
                                profile_key = "default_model_web"
                            elif ct == "qq":
                                profile_key = "default_model_qq"

                        if profile_key:
                            profile_id = str(conf().get(profile_key, "") or "").strip()
                            if profile_id:
                                custom_models = conf().get("custom_models", []) or []
                                if isinstance(custom_models, list):
                                    profile = next((m for m in custom_models if isinstance(m, dict) and m.get("id") == profile_id), None)
                                    if profile:
                                        model_override = {
                                            "model": profile.get("model") or "",
                                            "api_key": profile.get("api_key") or "",
                                            "api_base": profile.get("api_base") or "",
                                            "provider": profile.get("provider") or "custom",
                                            "profile_id": profile_id,
                                            "profile_key": profile_key,
                                        }
                                        context["model_override"] = model_override
                                        try:
                                            if hasattr(context, "kwargs") and isinstance(context.kwargs, dict):
                                                context.kwargs["model_override"] = model_override
                                        except Exception:
                                            pass
                                        logger.info(
                                            f"[Channel] Applied {profile_key} profile '{profile_id}' for channel_type="
                                            f"{context.get('channel_type', '')}, model={model_override.get('model', '')}, "
                                            f"provider={model_override.get('provider', '')}, agent_id={resolved_agent_id}, "
                                            f"session_key={context.get('session_key', '')}"
                                        )
                                    else:
                                        logger.warning(
                                            f"[Channel] {profile_key} points to missing profile_id='{profile_id}' "
                                            f"for channel_type={context.get('channel_type', '')}, agent_id={resolved_agent_id}"
                                        )
                            elif profile_key:
                                logger.info(
                                    f"[Channel] {profile_key} is empty for channel_type={context.get('channel_type', '')}, "
                                    f"agent_id={resolved_agent_id}; using main model={conf().get('model', '')}"
                                )
                    else:
                        existing_override = context.get("model_override") or {}
                        logger.info(
                            f"[Channel] Using explicit model_override for channel_type={context.get('channel_type', '')}, "
                            f"agent_id={resolved_agent_id}, model={existing_override.get('model', '')}, "
                            f"provider={existing_override.get('provider', '')}, "
                            f"profile_id={existing_override.get('profile_id', '')}"
                        )

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
                override = (context.get("model_override") if context else None) or {}
                logger.error(
                    "[Channel] Agent mode failed, fallback to normal mode: "
                    f"error={e}, channel_type={context.get('channel_type', '') if context else ''}, "
                    f"agent_id={context.get('agent_id', '') if context else ''}, "
                    f"session_key={context.get('session_key', '') if context else ''}, "
                    f"override_model={override.get('model', '')}, "
                    f"override_provider={override.get('provider', '')}, "
                    f"override_profile={override.get('profile_id', '')}, "
                    f"override_key={override.get('profile_key', '')}, "
                    f"fallback_main_model={conf().get('model', '')}",
                    exc_info=True,
                )
                # Fallback to normal mode if agent fails
                return Bridge().fetch_reply_content(query, context)
        else:
            # Normal mode
            return Bridge().fetch_reply_content(query, context)

    def build_voice_to_text(self, voice_file) -> Reply:
        return Bridge().fetch_voice_to_text(voice_file)

    def build_text_to_voice(self, text) -> Reply:
        return Bridge().fetch_text_to_voice(text)
