import json
import os
import traceback

from openai import OpenAI


api_key = "replace-with-client-api-key"
router_base = "http://prox1y.top:8080/v1/"
print(router_base)
client = OpenAI(api_key=api_key, base_url=router_base, timeout=30.0)

print("sdk_call_start")
try:
    resp = client.chat.completions.create(
        model="test",
        messages=[{"role": "user", "content": "which llm model you are?"}],
        stream=False,
    )
    content = resp.choices[0].message.content if resp.choices else ""
    content = content or ""
    print("sdk_call_ok", bool(resp.choices), content)
except Exception as exc:
    print("sdk_call_err", type(exc).__name__, str(exc))
    traceback.print_exc()
