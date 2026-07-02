"""Xiaomi MiMo provider profile."""

from providers import register_provider
from providers.base import ProviderProfile


class XiaomiProfile(ProviderProfile):
    """Xiaomi MiMo — supports reasoning toggle via chat_template_kwargs."""

    def build_extra_body(self, *, session_id=None, **context):
        """Inject chat_template_kwargs for mimo-v2.5 reasoning control.

        Reasoning config comes from reasoning_config context:
        - {"enabled": False} → enable_thinking=False (fast, 1-3s)
        - {"enabled": True, "effort": "low"/"medium"/"high"} → enable_thinking=True
        """
        extra = {}
        reasoning_config = context.get("reasoning_config") or {}

        # Default: enable thinking (full reasoning)
        enable_thinking = True

        if isinstance(reasoning_config, dict):
            if reasoning_config.get("enabled") is False:
                enable_thinking = False

        extra["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        return extra


xiaomi = XiaomiProfile(
    name="xiaomi",
    aliases=("mimo", "xiaomi-mimo"),
    env_vars=("XIAOMI_API_KEY",),
    base_url="https://api.xiaomimimo.com/v1",
    supports_health_check=False,  # /v1/models returns 401 even with valid key
    supports_vision=True,  # mimo-v2-omni is vision-capable
    supports_vision_tool_messages=False,  # rejects list-type tool content (400 "text is not set")
)

register_provider(xiaomi)
