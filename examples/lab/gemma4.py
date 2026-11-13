"""A tinker-cookbook renderer for Gemma 4, which the cookbook has none for.

Prompts are written by the model's own chat template, and a sampled turn is read by transformers' parser with the model's
response template. A sampled turn stays in the conversation as the tokens that were sampled: written again from its parsed
message it would differ in whitespace, the next prompt would no longer extend this one, and an episode would train as one
sequence per turn instead of one.
"""

import json
from pathlib import Path

import tinker
from huggingface_hub import hf_hub_download
from tinker_cookbook.renderers.base import (
  Message,
  ParseTermination,
  Renderer,
  TextPart,
  ThinkingPart,
  ToolCall,
  ToolSpec,
)
from transformers.utils.chat_parsing import parse_response

SAMPLED = "sampled_tokens"  # on a parsed turn: its tokens as they were sampled


class Gemma4(Renderer):
  def __init__(self, tokenizer, model: str, think: bool = True):
    super().__init__(tokenizer)
    self.think = think
    self.response = json.loads(Path(hf_hub_download(model, "tokenizer_config.json")).read_text())["response_template"]
    self.stop = [tokenizer.convert_tokens_to_ids(t) for t in ("<turn|>", "<|tool_response>")]
    # what the template writes to open a model turn, which a sampled turn's text follows
    user = [{"role": "user", "content": "-"}]
    self.opening = tokenizer.decode(self.template(user, add_generation_prompt=True)[len(self.template(user)) :])

  @property
  def has_extension_property(self) -> bool:
    return True

  def get_stop_sequences(self) -> list[int]:
    return self.stop

  def render_message(self, message, ctx):
    raise NotImplementedError("the chat template writes a conversation whole: build_generation_prompt")

  def create_conversation_prefix_with_tools(self, tools: list[ToolSpec], system_prompt: str = "") -> list[Message]:
    # the template declares the tools inside the system turn, so they ride on the system message
    return [{"role": "system", "content": system_prompt, "tools": tools}]

  def template(self, messages: list[dict], **kw) -> list[int]:
    return self.tokenizer.apply_chat_template(
      messages, tokenize=True, return_dict=False, enable_thinking=self.think, preserve_thinking=True, **kw
    )

  def to_openai_message(self, message: Message) -> dict:
    plain = super().to_openai_message(message)
    for call in plain.get("tool_calls", []):
      call["function"]["arguments"] = json.loads(call["function"]["arguments"])  # the template reads them as a dict
    # a sampled turn is written from its tokens; the template is only asked what follows its tool calls
    return plain | {"content": ""} if SAMPLED in message else plain

  def build_generation_prompt(self, messages: list[Message], role: str = "assistant", prefill: str | None = None):
    if role != "assistant" or prefill:
      raise NotImplementedError("only a model turn, without a prefill")
    tools = [{"type": "function", "function": t} for t in messages[0].get("tools", [])] if messages else []
    plain = [self.to_openai_message(m) for m in messages]
    sampled = [i for i, m in enumerate(messages) if SAMPLED in m]
    tokens = self.template(plain[: sampled[0] if sampled else None], tools=tools or None, add_generation_prompt=True)
    for i, following in zip(sampled, [*sampled[1:], len(messages)]):
      tokens += messages[i][SAMPLED]
      tokens += self.after(plain[i], plain[i + 1 : following], tokens[-1])
    return tinker.ModelInput.from_ints(tokens)

  def after(self, said: dict, replies: list[dict], last: int) -> list[int]:
    """The tokens that follow a sampled turn: the replies to it and the next turn's opening, as the template writes them."""
    turn, after = self.template([said]), self.template([said, *replies], add_generation_prompt=True)
    if after[: len(turn)] != turn:
      raise ValueError("the template rewrote a turn when its replies were added")
    # sampling stops at the stop token that ends the turn, with or without returning it
    stop = max(i for i, t in enumerate(turn) if t in self.stop)
    return ([] if last == turn[stop] else [turn[stop]]) + after[stop + 1 :]

  def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
    text = self.tokenizer.decode(response, skip_special_tokens=False)
    message: Message = {"role": "assistant", "content": text, SAMPLED: list(response)}
    # after tool replies the template has already opened the thinking channel
    opened = "<channel|>" in text and not text.startswith("<|channel>")
    try:
      said = parse_response(text, self.response, prefix=self.opening + ("<|channel>thought\n" if opened else ""))
    except ValueError:
      return message, ParseTermination.MALFORMED
    calls = said.get("tool_calls") or []
    if not calls and "<|tool_call>" in text:
      return message, ParseTermination.MALFORMED
    thinking = said.get("thinking") or said.get("reasoning_content") or ""
    message["content"] = [ThinkingPart(type="thinking", thinking=thinking)] if thinking else []
    message["content"].append(TextPart(type="text", text=said.get("content") or ""))
    if calls:
      bodies = [
        ToolCall.FunctionBody(name=c["function"]["name"], arguments=json.dumps(c["function"]["arguments"]))
        for c in calls
      ]
      message["tool_calls"] = [ToolCall(id=f"call_{n}", function=body) for n, body in enumerate(bodies)]
    return message, ParseTermination.STOP_SEQUENCE if response[-1] in self.stop else ParseTermination.EOS
