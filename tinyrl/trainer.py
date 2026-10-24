# a trainer serves one pool through the bus: the lora adapters that share a base model, or a single full fine-tune.
from __future__ import annotations
import asyncio, glob, json, math, os, socket, sys, time
from typing import Callable
import numpy as np, safetensors.torch, torch
from peft import LoraConfig, set_peft_model_state_dict
from torch.utils.checkpoint import checkpoint
from transformers import AutoModelForCausalLM, PreTrainedModel
from tinyrl import wire
from tinyrl.bus import Bus, Cmd, Reply
from tinyrl.helpers import DEBUG, getenv

TOKEN_BUDGET, CHUNK, GRAD_CKPT = getenv("TOKEN_BUDGET", 16384), getenv("CHUNK", 1024), getenv("GRAD_CKPT", 1)

# ***** losses, elementwise over target tokens. lp is the policy's logprob, old the sampler's, adv the advantage, w the weight *****

def ratio(lp, old): return (lp-old).clamp(-20, 20).exp()
def ppo(lp, w, old, adv, cfg):
  r, eps = ratio(lp, old), cfg.get("clip_range", 0.2)
  return -(torch.min(r*adv, r.clamp(1-eps, 1+eps)*adv) - cfg.get("kl_coeff", 0.0)*(r-1-(lp-old).clamp(-20, 20)))*w
LOSSES = {"cross_entropy": lambda lp, w, old, adv, cfg: -lp*w, "ppo": ppo,
          "importance_sampling": lambda lp, w, old, adv, cfg: torch.nan_to_num(-ratio(lp, old)*adv*w, nan=0.0, posinf=0.0, neginf=0.0)}

# ***** forward *****

def batches(lens:list[int], budget:int) -> list[list[int]]:
  # shortest first so a pass pads little. a sequence starts a new pass if joining would pass the budget or make the pass >25% padding
  out:list[list[int]] = []
  for i in sorted(range(len(lens)), key=lens.__getitem__):
    if out and (n:=lens[i]*(len(out[-1])+1)) <= budget and n - lens[i] - sum(lens[j] for j in out[-1]) <= n/4: out[-1].append(i)
    else: out.append([i])
  return out

def pad(xs:list[np.ndarray], lens:list[int], dtype, device:torch.device) -> torch.Tensor:
  out = np.zeros((len(xs), max(lens)), dtype)
  for i,(x,n) in enumerate(zip(xs, lens)): out[i, :n] = x[:n]
  return torch.from_numpy(out).to(device)

def logprobs(lm:PreTrainedModel, ids:torch.Tensor, mask:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
  # the output head and logsumexp run CHUNK positions at a time under checkpoint, so [tokens, vocab] logits never exist at once
  h = lm(ids, attention_mask=mask, use_cache=False, logits_to_keep=1, output_hidden_states=True).hidden_states[-1].flatten(0, 1)
  head, cap = lm.get_output_embeddings(), getattr(lm.config.get_text_config(), "final_logit_softcapping", None)
  def chunk(h:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
    logits = (torch.tanh(head(h)/cap)*cap if cap else head(h)).float()
    return logits.gather(-1, t[:, None])[:, 0] - logits.logsumexp(-1)
  return torch.cat([checkpoint(chunk, *c, use_reentrant=False) for c in zip(h.split(CHUNK), targets.flatten().split(CHUNK))]).view(targets.shape)

# ***** model *****

def load(name:str) -> PreTrainedModel:
  gpu = torch.cuda.is_available()
  return AutoModelForCausalLM.from_pretrained(name, dtype=torch.bfloat16 if gpu else torch.float32, device_map="auto" if gpu else None)

class Model:
  # a base model and the runs training on it: a lora adapter per run, or the whole model for a single full fine-tune
  def __init__(self, load:Callable[[str], PreTrainedModel]=load):
    self.load, self.base, self.lora = load, "", True
    self.lm:PreTrainedModel|None = None
    self.params:dict[str, list[torch.nn.Parameter]] = {}
    self.opts:dict[str, torch.optim.AdamW] = {}

  def create(self, run:str, base:str, rank:int, cfg:dict) -> None:
    if self.lm is None:
      self.lm, self.base, self.lora = self.load(base), base, rank > 0
      if GRAD_CKPT: self.lm.gradient_checkpointing_enable()
      if GRAD_CKPT and self.lora: self.lm.enable_input_require_grads()  # with the embeddings frozen, checkpointed blocks need a grad path
    if (base, rank > 0) != (self.base, self.lora): raise ValueError(f"this trainer does {'lora' if self.lora else 'full'} fine-tuning of {self.base}")
    if cfg.get("seed") is not None: torch.manual_seed(cfg["seed"])
    if not self.lora:
      if self.params: raise ValueError(f"a full fine-tune has its trainer to itself, and {next(iter(self.params))} is on this one")
      self.params[run] = [p.requires_grad_() for p in self.lm.parameters()]
      return
    targets = ["q_proj", "k_proj", "v_proj", "o_proj"]*cfg.get("train_attn", True) + ["gate_proj", "up_proj", "down_proj"]*cfg.get("train_mlp", True)
    # an adapter on a tied head can't be loaded by vllm
    targets += ["lm_head"]*(cfg.get("train_unembed", True) and not getattr(self.lm.config.get_text_config(), "tie_word_embeddings", False))
    lora = LoraConfig(r=rank, lora_alpha=cfg.get("lora_alpha", 16), lora_dropout=cfg.get("lora_dropout", 0.05), target_modules=targets)
    self.lm.add_adapter(lora, adapter_name=run)
    self.params[run] = [p for n,p in self.lm.named_parameters() if f".{run}." in n]

  def use(self, run:str) -> PreTrainedModel:
    if self.lm is None or run not in self.params: raise KeyError(f"model {run} is not on this trainer, it was deleted or its trainer restarted")
    if self.lora: self.lm.set_adapter(run)
    return self.lm

  def forward_backward(self, run:str, data:list[wire.Datum], loss_fn:str, cfg:dict, forward_only:bool) -> tuple[list[np.ndarray], dict[str, float]]:
    if loss_fn not in LOSSES: raise ValueError(f"unknown loss_fn {loss_fn}, this trainer has {', '.join(LOSSES)}")
    lm, lens, total = self.use(run).train(not forward_only), [min(len(d.tokens), len(d.inputs["target_tokens"])) for d in data], 0.0
    out:list[np.ndarray] = [np.zeros(0, np.float32)]*len(data)
    with torch.set_grad_enabled(not forward_only):
      for b in batches(lens, TOKEN_BUDGET):
        ds, ls = [data[i] for i in b], [lens[i] for i in b]
        def col(k:str, fill:float=0.0, dtype=np.float32) -> torch.Tensor:
          return pad([d.inputs.get(k, np.full(n, fill)) for d,n in zip(ds, ls)], ls, dtype, lm.device)
        ids, mask = pad([d.tokens for d in ds], ls, np.int64, lm.device), pad([np.ones(n) for n in ls], ls, np.int64, lm.device)
        lp = logprobs(lm, ids, mask, col("target_tokens", dtype=np.int64))
        loss = LOSSES[loss_fn](lp, col("weights", 1.0), col("logprobs"), col("advantages"), cfg).sum()
        if not forward_only: loss.backward()
        total += loss.item()
        # finite, so json can carry it
        for r,i in enumerate(b): out[i] = torch.nan_to_num(lp[r, :lens[i]].detach().float(), nan=0.0).clamp(-9999, 9999).cpu().numpy()
    return out, {"loss:sum": total, "loss:mean": total/max(1, len(data))}

  def opt(self, run:str) -> torch.optim.AdamW:
    if run not in self.opts: self.opts[run] = torch.optim.AdamW(self.params[run])
    return self.opts[run]

  def optim_step(self, run:str, adam:dict) -> dict[str, float]:
    self.use(run)
    hp = {"lr": adam.get("learning_rate", 1e-4), "betas": (adam.get("beta1", 0.9), adam.get("beta2", 0.95)), "eps": adam.get("eps", 1e-12),
          "weight_decay": adam.get("weight_decay", 0.0)}
    for g in (opt:=self.opt(run)).param_groups: g.update(hp)  # every step brings its own hyperparameters, so a schedule just works
    norm = torch.nn.utils.clip_grad_norm_(self.params[run], adam.get("grad_clip_norm") or math.inf).item()
    opt.step()
    opt.zero_grad()
    return {"grad_norm:mean": norm}

  def save(self, run:str, path:str, optimizer:bool) -> None:
    self.use(run).save_pretrained(path)  # an adapter in peft format or the whole model, either way what vllm loads
    if optimizer: torch.save(self.opt(run).state_dict(), f"{path}/optimizer.pt")

  def load_weights(self, run:str, path:str, optimizer:bool) -> None:
    files = glob.glob(f"{path}/*.safetensors")
    lm, sd = self.use(run), {k.removeprefix("base_model.model."):v for f in files for k,v in safetensors.torch.load_file(f).items()}
    if not sd: raise ValueError(f"no weights at {path}")
    if optimizer and not os.path.exists(f"{path}/optimizer.pt"): raise ValueError(f"no optimizer state at {path}")
    res = set_peft_model_state_dict(lm, sd, adapter_name=run) if self.lora else lm.load_state_dict(sd, strict=False)
    if res.unexpected_keys: raise ValueError(f"{path} does not fit this model: {', '.join(res.unexpected_keys[:3])}")
    self.opts.pop(run, None)
    if optimizer: self.opt(run).load_state_dict(torch.load(f"{path}/optimizer.pt", map_location="cpu"))

  def delete(self, run:str) -> None:
    lm = self.use(run)
    if self.lora: lm.delete_adapter(run)
    del self.params[run]
    self.opts.pop(run, None)

# ***** trainer *****

def ok(d:dict) -> Reply: return Reply(True, wire.JSON, json.dumps(d).encode())
def fail(e:Exception) -> Reply:
  # a request the trainer can't take is the user's error, anything else is ours
  err = {"error": f"{type(e).__name__}: {e}", "category": "user" if isinstance(e, (ValueError, KeyError)) else "server"}
  return Reply(False, wire.JSON, json.dumps(err).encode())

class Trainer:
  def __init__(self, bus:Bus, pool:str, model:Model|None=None, worker:str=""):
    self.bus, self.pool, self.model, self.worker, self.done = bus, pool, model or Model(), worker or f"{socket.gethostname()}-{os.getpid()}", False

  def run(self, c:Cmd) -> Reply:
    m = self.model
    try:
      if c.op == "forward_backward":
        req = wire.parse(c.body)
        lps, metrics = m.forward_backward(c.run, wire.datums(req), req.loss_fn or "cross_entropy", wire.config(req), req.forward_only)
        return Reply(True, wire.PROTO, wire.output(lps, metrics))
      a = json.loads(c.body)
      match c.op:
        case "create_model": m.create(c.run, a["base_model"], a["rank"], a["lora"])
        case "optim_step": return ok({"type": "optim_step", "metrics": m.optim_step(c.run, a["adam_params"])})
        case "save_weights" | "save_weights_for_sampler": m.save(c.run, a["path"], optimizer=c.op == "save_weights")
        case "load_weights": m.load_weights(c.run, a["path"], a["optimizer"])
        case "delete_model": m.delete(c.run)
        case _: raise ValueError(f"unknown op {c.op}")
      return ok({"type": c.op, "model_id": c.run} | ({"path": a["uri"]} if "uri" in a else {}))
    except Exception as e: return fail(e)

  async def step(self, count:int=16, block:int=1000) -> int:
    for c in (cmds:=await self.bus.take(self.pool, self.worker, count, block)):
      st, rep = time.perf_counter(), await asyncio.to_thread(self.run, c)
      await self.bus.finish(c, rep)
      if c.op == "delete_model" and rep.ok:
        await self.bus.drop(c.run, self.pool)
        self.done = not self.model.lora  # a full fine-tune's trainer exits with its run
      if DEBUG: print(f"{self.pool} {c.rid} {c.op:24s} {(time.perf_counter()-st)*1e3:9.2f} ms" + ("" if rep.ok else f"  {rep.body.decode()}"))
    return len(cmds)

  async def serve(self) -> None:
    for c in await self.bus.reclaim(self.pool, self.worker): await self.bus.finish(c, fail(RuntimeError(f"its trainer died during {c.op}")))
    while not self.done: await self.step()

if __name__ == "__main__": asyncio.run(Trainer(Bus(), sys.argv[1]).serve())
