with open("tinyrl/workers/trainer.py") as f:
  text = f.read()

text = text.replace(
  """  async def restore_missing_run(self, cmd: Cmd) -> None:
    \"\"\"Bring back a run this trainer doesn't hold: from the state its last trainer saved when it stopped, or empty, for load_weights to fill. A run whose trainer died without saving it can't go on from where it was.\"\"\"
    run, uri = await self.bus.get_run(cmd.run), await self.bus.parked_checkpoint(cmd.run)
    if not uri and cmd.op not in (Op.LOAD_WEIGHTS, Op.DELETE_MODEL):
      raise KeyError(
        f"model {cmd.run} lost its unsaved steps when its trainer stopped. load_state to go on from a checkpoint"
      )
    await asyncio.to_thread(self.model.create_run, cmd.run, run["base"], run["rank"], run.get("lora", {}))
    if uri:
      version = (await self.registry.get_checkpoint(uri) or {}).get("version")
      tensors, meta = await asyncio.to_thread(self.store.get, uri, version)
      await asyncio.to_thread(self.model.restore_run, cmd.run, tensors, meta, True)
      if await self.bus.set_parked_checkpoint(cmd.run, None, self.worker, self.owner):
        await self.registry.unregister_checkpoints(uri)  # the node daemons' sweeps delete it""",
  """  async def restore_missing_run(self, cmd: Cmd) -> None:
    \"\"\"Bring back a run this trainer doesn't hold: from the state its last trainer saved when it stopped, or empty, for load_weights to fill. A run whose trainer died without saving it can't go on from where it was.\"\"\"
    run, uri = await self.bus.get_run(cmd.run), await self.bus.parked_checkpoint(cmd.run)
    if not uri and cmd.op not in (Op.LOAD_WEIGHTS, Op.DELETE_MODEL):
      raise KeyError(
        f"model {cmd.run} lost its unsaved steps when its trainer stopped. load_state to go on from a checkpoint"
      )
    await asyncio.to_thread(self.model.create_run, cmd.run, run["base"], run["rank"], run.get("lora", {}))
    try:
      if uri:
        version = (await self.registry.get_checkpoint(uri) or {}).get("version")
        tensors, meta = await asyncio.to_thread(self.store.get, uri, version)
        await asyncio.to_thread(self.model.restore_run, cmd.run, tensors, meta, True)
        if await self.bus.set_parked_checkpoint(cmd.run, None, self.worker, self.owner):
          await self.registry.unregister_checkpoints(uri)  # the node daemons' sweeps delete it
    except Exception:
      await asyncio.to_thread(self.model.delete_run, cmd.run)
      raise""",
)

text = text.replace(
  """    try:
      if cmd.op != Op.CREATE_MODEL and cmd.run not in self.model.params:
        await self.restore_missing_run(cmd)
    except Exception as e:
      return fail(e)
    version = None
    if cmd.op == Op.LOAD_WEIGHTS:  # the version saved last, which may be another node's, still uploading
      version = (await self.registry.get_checkpoint(json.loads(cmd.body)["uri"]) or {}).get("version")
    # off the event loop, so redis stays responsive
    reply, size = await asyncio.to_thread(self.run_command, cmd, version)""",
  """    created_here = False
    try:
      if cmd.op != Op.CREATE_MODEL and cmd.run not in self.model.params:
        await self.restore_missing_run(cmd)
        created_here = True
    except Exception as e:
      return fail(e)
    version = None
    if cmd.op == Op.LOAD_WEIGHTS:  # the version saved last, which may be another node's, still uploading
      version = (await self.registry.get_checkpoint(json.loads(cmd.body)["uri"]) or {}).get("version")
    # off the event loop, so redis stays responsive
    reply, size = await asyncio.to_thread(self.run_command, cmd, version)
    if not reply.ok and created_here:
      await asyncio.to_thread(self.model.delete_run, cmd.run)""",
)

with open("tinyrl/workers/trainer.py", "w") as f:
  f.write(text)

with open("tinyrl/workers/model.py") as f:
  text = f.read()

text = text.replace(
  """    self.pinned: dict[int, torch.Tensor] = {}
    self.parked = False
    self.dirty: set[str] = set()""",
  """    self.pinned: dict[int, torch.Tensor] = {}
    self.parked = False
    self.dirty: set[str] = set()
    self.needs_recovery: set[str] = set()""",
)

text = text.replace(
  """  def delete_run(self, run: str) -> None:
    lm = self.activate_run(run)
    self.dirty.discard(run)""",
  """  def delete_run(self, run: str) -> None:
    lm = self.activate_run(run)
    self.dirty.discard(run)
    self.needs_recovery.discard(run)""",
)

text = text.replace(
  """  def forward_backward(self, run: str, data: list[wire.Datum], loss_fn: str, config: dict, forward_only: bool):
    \"\"\"Run the loss over every datum, accumulating gradients unless forward_only. Returns each datum's target logprobs, and the loss metrics.\"\"\"
    if loss_fn not in LOSSES:
      raise ValueError(f"unknown loss_fn {loss_fn}, this trainer has {', '.join(LOSSES)}")

    lm = self.activate_run(run).train(not forward_only)
    lengths = [min(len(d.tokens), len(d.inputs["target_tokens"])) for d in data]
    logprobs: list[np.ndarray] = [np.zeros(0, np.float32)] * len(data)
    total = 0.0

    # on the gpu autocast runs the adapters' matmuls in bf16 anyway, so PEFT's upcast of each layer's input to the fp32 adapter only adds casts
    gpu = lm.device.type == "cuda"
    with torch.set_grad_enabled(not forward_only), disable_input_dtype_casting(lm, active=gpu):
      for batch in batches(lengths, token_budget(lm)):
        x = batch_tensors([data[i] for i in batch], [lengths[i] for i in batch], lm.device)
        with torch.autocast(lm.device.type, dtype=torch.bfloat16, enabled=gpu):
          lp = target_logprobs(lm, x["ids"], x["mask"], x["targets"])
        loss = LOSSES[loss_fn](lp, x["weights"], x["sampler_logprobs"], x["advantages"], config).sum()
        if not forward_only:
          loss.backward()
        total += loss.item()

        for row, i in enumerate(batch):
          values = lp[row, : lengths[i]].detach().float()
          # keep them finite, so a JSON client can carry them
          logprobs[i] = torch.nan_to_num(values, nan=0.0).clamp(-9999, 9999).cpu().numpy()

    return logprobs, {"loss:sum": total, "loss:mean": total / max(1, len(data))}""",
  """  def forward_backward(self, run: str, data: list[wire.Datum], loss_fn: str, config: dict, forward_only: bool):
    \"\"\"Run the loss over every datum, accumulating gradients unless forward_only. Returns each datum's target logprobs, and the loss metrics.\"\"\"
    if loss_fn not in LOSSES:
      raise ValueError(f"unknown loss_fn {loss_fn}, this trainer has {', '.join(LOSSES)}")

    self.needs_recovery.discard(run)
    lm = self.activate_run(run).train(not forward_only)
    lengths = [min(len(d.tokens), len(d.inputs["target_tokens"])) for d in data]
    logprobs: list[np.ndarray] = [np.zeros(0, np.float32)] * len(data)
    total = 0.0

    # on the gpu autocast runs the adapters' matmuls in bf16 anyway, so PEFT's upcast of each layer's input to the fp32 adapter only adds casts
    gpu = lm.device.type == "cuda"
    try:
      with torch.set_grad_enabled(not forward_only), disable_input_dtype_casting(lm, active=gpu):
        for batch in batches(lengths, token_budget(lm)):
          x = batch_tensors([data[i] for i in batch], [lengths[i] for i in batch], lm.device)
          with torch.autocast(lm.device.type, dtype=torch.bfloat16, enabled=gpu):
            lp = target_logprobs(lm, x["ids"], x["mask"], x["targets"])
          loss = LOSSES[loss_fn](lp, x["weights"], x["sampler_logprobs"], x["advantages"], config).sum()
          if not forward_only:
            loss.backward()
          total += loss.item()

          for row, i in enumerate(batch):
            values = lp[row, : lengths[i]].detach().float()
            # keep them finite, so a JSON client can carry them
            logprobs[i] = torch.nan_to_num(values, nan=0.0).clamp(-9999, 9999).cpu().numpy()

      return logprobs, {"loss:sum": total, "loss:mean": total / max(1, len(data))}
    except Exception:
      if not forward_only and total > 0:
        self.needs_recovery.add(run)
      raise""",
)

text = text.replace(
  """  def optim_step(self, run: str, adam: dict) -> dict[str, float]:
    self.activate_run(run)
    optimizer = self.adam_for_run(run)""",
  """  def optim_step(self, run: str, adam: dict) -> dict[str, float]:
    if run in getattr(self, "needs_recovery", set()):
      raise RuntimeError("run needs recovery")
    self.activate_run(run)
    optimizer = self.adam_for_run(run)""",
)

text = text.replace(
  """  def training_state(self, run: str) -> tuple[dict[str, torch.Tensor], dict]:
    \"\"\"A run's portable training checkpoint: fp32 master weights and named AdamW state, with no pickle.\"\"\"
    self.activate_run(run)
    tensors = {name: p.detach() for name, p in self.params[run].items()}
    optim = self.optimizers.get(run)
    for name, p in self.params[run].items():
      if optim and p in optim.state:
        s = optim.state[p]
        tensors[f"{name}.exp_avg"] = s["exp_avg"].detach()
        tensors[f"{name}.exp_avg_sq"] = s["exp_avg_sq"].detach()
        tensors[f"{name}.step"] = s["step"].detach()
    return tensors, {"optimizer": True}""",
  """  def training_state(self, run: str) -> tuple[dict[str, torch.Tensor], dict]:
    \"\"\"A run's portable training checkpoint: fp32 master weights and named AdamW state, with no pickle.\"\"\"
    self.activate_run(run)
    tensors = {name: p.detach() for name, p in self.params[run].items()}
    optim = self.optimizers.get(run)
    for name, p in self.params[run].items():
      if p.grad is not None:
        tensors[f"{name}.grad"] = p.grad.detach()
      if optim and p in optim.state:
        s = optim.state[p]
        tensors[f"{name}.exp_avg"] = s["exp_avg"].detach()
        tensors[f"{name}.exp_avg_sq"] = s["exp_avg_sq"].detach()
        tensors[f"{name}.step"] = s["step"].detach()
    return tensors, {"optimizer": True, "internal_handoff": True}""",
)

text = text.replace(
  """  def restore_run(self, run: str, tensors: dict[str, torch.Tensor], meta: dict, optimizer: bool) -> None:
    self.activate_run(run)
    if optimizer and not meta.get("optimizer"):
      raise ValueError("checkpoint has no optimizer state")
    suffixes = (".exp_avg", ".exp_avg_sq", ".step")
    weights = {k.removeprefix("base_model.model."): v for k, v in tensors.items() if not k.endswith(suffixes)}
    if not self.lora and "lm_head.weight" not in self.params[run]:
      weights.pop("lm_head.weight", None)  # tied to embed_tokens
    if weights.keys() != self.params[run].keys():
      raise ValueError("checkpoint does not fit this model")
    for name, p in self.params[run].items():
      p.data.copy_(weights[name])

    self.pinned.clear()""",
  """  def restore_run(self, run: str, tensors: dict[str, torch.Tensor], meta: dict, optimizer: bool) -> None:
    self.activate_run(run)
    if optimizer and not meta.get("optimizer"):
      raise ValueError("checkpoint has no optimizer state")
    suffixes = (".exp_avg", ".exp_avg_sq", ".step", ".grad")
    weights = {k.removeprefix("base_model.model."): v for k, v in tensors.items() if not k.endswith(suffixes)}
    if not self.lora and "lm_head.weight" not in self.params[run]:
      weights.pop("lm_head.weight", None)  # tied to embed_tokens
    if weights.keys() != self.params[run].keys():
      raise ValueError("checkpoint does not fit this model")
    for name, p in self.params[run].items():
      if weights[name].shape != p.shape:
        raise ValueError(f"shape mismatch for {name}")
    if optimizer and any(k.endswith(".step") for k in tensors):
      for name in self.params[run]:
        if not all(f"{name}.{s}" in tensors for s in ("exp_avg", "exp_avg_sq", "step")):
          raise ValueError("missing opt")

    for name, p in self.params[run].items():
      p.data.copy_(weights[name])
      if meta.get("internal_handoff") and f"{name}.grad" in tensors:
        p.grad = tensors[f"{name}.grad"].to(p.device)
      else:
        p.grad = None

    self.pinned.clear()""",
)

with open("tinyrl/workers/model.py", "w") as f:
  f.write(text)
