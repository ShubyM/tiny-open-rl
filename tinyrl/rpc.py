"""Internal single-RPC HTTP transport: `POST /internal/exchange`."""

from __future__ import annotations

import base64
import json
from dataclasses import fields

from tinyrl.control.accounting import Charge
from tinyrl.helpers import getenv
from tinyrl.types import Cmd, NodeReply, NodeReport, Op, Reply, Role, Turn, TurnResult, WorkerSpec, WorkerStatus

WIRE = {c.__name__: c for c in (Charge, Cmd, NodeReply, NodeReport, Reply, Turn, TurnResult, WorkerSpec, WorkerStatus)}
TUPLES = {"gpus", "bound", "loaded"}


def encode_wire(val):
  if isinstance(val, bytes):
    return {"__bytes__": base64.b64encode(val).decode("ascii")}
  if (name := type(val).__name__) in WIRE:
    return {"__type__": name, **{f.name: getattr(val, f.name) for f in fields(val)}}
  raise TypeError(f"cannot serialize {type(val).__name__}")


def decode_wire(obj: dict):
  if "__bytes__" in obj:
    return base64.b64decode(obj["__bytes__"])
  if (name := obj.get("__type__")) not in WIRE:
    return obj
  d = {k: tuple(v) if k in TUPLES and isinstance(v, list) else v for k, v in obj.items() if k != "__type__"}
  if name == "Cmd":
    d["op"] = Op(d["op"])
  elif name == "WorkerSpec":
    d["role"] = Role(d["role"])
  elif name == "TurnResult" and d.get("samples"):
    d["samples"] = [tuple(x) for x in d["samples"]]
  return WIRE[name](**d)


async def serve_control(bus, host: str = "0.0.0.0", port: int = 8001):
  """Serve the controller's single `exchange` RPC over HTTP. Returns `(runner, url)`."""
  from aiohttp import web

  async def handle(req: web.Request) -> web.Response:
    reply = await bus.exchange(
      json.loads(await req.read(), object_hook=decode_wire), int(req.query.get("wait_ms", "0"))
    )
    return web.Response(body=json.dumps(reply, default=encode_wire).encode(), content_type="application/json")

  app = web.Application(client_max_size=128 * 1024 * 1024)
  app.router.add_post("/internal/exchange", handle)
  runner = web.AppRunner(app, access_log=None)
  await runner.setup()
  site = web.TCPSite(runner, host, port)
  await site.start()
  return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


class ControlClient:
  """Async HTTP client for the controller's single `exchange` RPC."""

  def __init__(self, url: str | None = None):
    self.url = (getenv("CONTROL_URL", "http://127.0.0.1:8001") if url is None else url).rstrip("/")
    self.session = None

  async def aclose(self) -> None:
    if self.session and not self.session.closed:
      await self.session.close()
      self.session = None

  async def exchange(self, report: NodeReport, wait_ms: int = 0) -> NodeReply:
    import aiohttp

    if self.session is None or self.session.closed:
      self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    t = aiohttp.ClientTimeout(total=max(5.0, wait_ms / 1000 + 5.0))
    body = json.dumps(report, default=encode_wire).encode()
    async with self.session.post(
      f"{self.url}/internal/exchange", params={"wait_ms": str(wait_ms)}, data=body, timeout=t
    ) as r:
      if r.status >= 400:
        raise aiohttp.ClientError(f"control server returned {r.status}")
      return json.loads(await r.read(), object_hook=decode_wire)
