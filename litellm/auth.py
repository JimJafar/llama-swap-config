"""API key -> Opik project, without LiteLLM's key database.

An agent picks its Opik project through the API key it sends:
`opik-<project>` (e.g. `opik-hermes`) logs to project `<project>`. Any other key,
or none, is accepted and logs to `unlabelled`. Nothing is secret: the gateway is
tailnet-only, like llama-swap behind it.
"""
import re

from fastapi import Request
from litellm.proxy._types import UserAPIKeyAuth

PROJECT_KEY = re.compile(r"opik-([A-Za-z0-9][A-Za-z0-9_.-]{0,63})")


async def user_api_key_auth(request: Request, api_key: str) -> UserAPIKeyAuth:
    m = PROJECT_KEY.fullmatch(api_key or "")
    project = m.group(1) if m else "unlabelled"
    return UserAPIKeyAuth(api_key=api_key or "none", key_alias=project,
                          metadata={"opik": {"project_name": project}})
