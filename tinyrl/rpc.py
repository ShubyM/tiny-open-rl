"""Internal HTTP JSON-RPC transport between remote node daemons and the control-plane Bus."""

from __future__ import annotations

import asyncio
import base64
import inspect
import json
from dataclasses import asdict

from tinyrl.control.accounting import Charge
from tinyrl.helpers import getenv
from tinyrl.types import Cmd, Event, Op, Reply, Role, Run

DATACLASSES = {"Cmd": Cmd, "Reply": Reply, "Charge": Charge, "Run": Run, "Event": Event}
NON_IDEMPOTENT = frozenset({"claim_trainer", "take_commands"})
LOCAL_ASYNC = frozenset({"aclose", "sweep_checkpoints", "peek"})


def error_types() -> dict[str, type[Exception]]:
  from tinyrl.bus import BackpressureError, PromiseGoneError

  return {cls.__name__: cls for cls in (ValueError, KeyError, RuntimeError, PromiseGoneError, BackpressureError)}


def encode_rpc(value):
  if isinstance(value, bytes):
    return {"__bytes__": base64.b64encode(value).decode("ascii")}
  if isinstance(value, (Cmd, Reply, Charge, Run, Event)):
    return {"__type__": type(value).__name__, **asdict(value)}
  if isinstance(value, Exception):
    return {"__error__": [type(value).__name__, str(value)]}
  raise TypeError(f"cannot serialize {type(value).__name__}")


def decode_rpc(obj: dict):
  if "__bytes__" in obj:
    return base64.b64decode(obj["__bytes__"])
  if "__error__" in obj:
    name, message = obj["__error__"]
    return error_types().get(name, RuntimeError)(message)
  if (name := obj.get("__type__")) in DATACLASSES:
    fields = {k: v for k, v in obj.items() if k != "__type__"}
    if name == "Cmd":
      fields["op"] = Op(fields["op"])
    elif name == "Run":
      fields["role"] = Role(fields["role"])
    return DATACLASSES[name](**fields)
  return obj


def dumps_rpc(value) -> bytes:
  return json.dumps(value, default=encode_rpc).encode()


def loads_rpc(raw: bytes):
  return json.loads(raw, object_hook=decode_rpc)


def rpc_methods(bus_cls) -> frozenset[str]:
  return frozenset(
    name for name, fn in bus_cls.__dict__.items() if inspect.iscoroutinefunction(fn) and name not in LOCAL_ASYNC
  )


async def serve_bus(bus, host: str = "0.0.0.0", port: int = 8001):
  """Serve Bus methods over internal HTTP POST `/bus/{name}`. Returns `(runner, url)`."""
  from aiohttp import web

  methods = rpc_methods(type(bus))

  async def handle(request: web.Request) -> web.Response:
    name = request.match_info["name"]
    if name not in methods:
      return web.Response(status=404)
    args, kwargs = loads_rpc(await request.read())
    try:
      result = await getattr(bus, name)(*args, **kwargs)
      return web.Response(body=dumps_rpc({"ok": result}), content_type="application/json")
    except Exception as exc:
      return web.Response(body=dumps_rpc({"err": exc}), status=400, content_type="application/json")

  app = web.Application(client_max_size=128 * 1024 * 1024)
  app.router.add_post("/bus/{name}", handle)
  runner = web.AppRunner(app, access_log=None)
  await runner.setup()
  site = web.TCPSite(runner, host, port)
  await site.start()
  return runner, f"http://127.0.0.1:{site._server.sockets[0].getsockname()[1]}"


class RemoteBus:
  """Async HTTP client exposing the Bus interface against a control server's `/bus/*` endpoint."""

  def __init__(self, url: str | None = None):
    self.url = (getenv("CONTROL_URL", "http://127.0.0.1:8001") if url is None else url).rstrip("/")
    self.session = None
    from tinyrl.bus import Bus

    self.methods = rpc_methods(Bus)

  async def aclose(self) -> None:
    if self.session and not self.session.closed:
      await self.session.close()
      self.session = None

  async def sweep_checkpoints(self, store, grace: float) -> None:
    unused = await asyncio.to_thread(store.versions)
    unused.difference_update(await self.live_versions())
    await asyncio.to_thread(store.sweep, unused, grace)

  async def call(self, name: str, *args, **kwargs):
    import aiohttp

    if self.session is None or self.session.closed:
      self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None))
    payload, delay = dumps_rpc([args, kwargs]), 0.1
    for attempt in range(30):
      try:
        async with self.session.post(f"{self.url}/bus/{name}", data=payload) as resp:
          if resp.status >= 500:
            raise aiohttp.ClientError(f"control server returned {resp.status}")
          msg = loads_rpc(await resp.read())
          if "err" in msg:
            raise msg["err"]
          return msg["ok"]
      except (aiohttp.ClientError, OSError) as exc:
        sent_non_idempotent = name in NON_IDEMPOTENT and not isinstance(exc, aiohttp.ClientConnectorError)
        if attempt == 29 or sent_non_idempotent:
          raise
        await asyncio.sleep(delay)
        delay = min(2.0, delay * 2)

  def __getattr__(self, name: str):
    if name in self.methods:
      return lambda *args, **kwargs: self.call(name, *args, **kwargs)
    raise AttributeError(name)
