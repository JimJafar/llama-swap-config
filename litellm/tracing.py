"""LiteLLM's Opik logger, minus embeddings.

An embedding trace is the input text and a 1024-number vector: no use for evals, and
an indexer would make a lot of them. Chat, audio and everything else is logged.
"""
from litellm.integrations.opik.opik import OpikLogger

SKIP = {"embedding", "aembedding"}


class ChatOnlyOpik(OpikLogger):
    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        if kwargs.get("call_type") not in SKIP:
            await super().async_log_success_event(kwargs, response_obj, start_time, end_time)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        if kwargs.get("call_type") not in SKIP:
            super().log_success_event(kwargs, response_obj, start_time, end_time)


opik = ChatOnlyOpik()
