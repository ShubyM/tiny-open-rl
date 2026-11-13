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
from tinker_cookbook.renderers.base import Message, ParseTermination, Renderer, ToolCall, ToolSpec
from transformers.utils.chat_parsing import parse_response

SAMPLED_TOKENS = "sampled_tokens"  # on a parsed turn: its tokens as they were sampled


class Gemma4Renderer(Renderer):
  def __init__(self, tokenizer, model: str, think: bool = True):
    super().__init__(tokenizer)
    self.template_kwargs = {"tokenize": True, "return_dict": False, "enable_thinking": think, "preserve_thinking": True}
    config = json.loads(Path(hf_hub_download(model, "tokenizer_config.json")).read_text())
    self.response_template = config["response_template"]
    self.stop_tokens = [tokenizer.convert_tokens_to_ids(t) for t in ("<turn|>", "<|tool_response>")]
    # what the template writes to open a model turn, which a sampled turn's text follows
    user = [{"role": "user", "content": "-"}]
    self.prefix = tokenizer.decode(self.render(user, add_generation_prompt=True)[len(self.render(user)) :])

  @property
  def has_extension_property(self) -> bool:
    return True

  def get_stop_sequences(self) -> list[int]:
    return self.stop_tokens

  def render_message(self, message, ctx):
    raise NotImplementedError("the chat template writes a conversation whole: build_generation_prompt")

  def create_conversation_prefix_with_tools(self, tools: list[ToolSpec], system_prompt: str = "") -> list[Message]:
    # the template declares the tools inside the system turn, so they are kept on the system message
    return [{"role": "system", "content": system_prompt, "tools": tools}]

  def render(self, messages: list[dict], **kw) -> list[int]:
    return self.tokenizer.apply_chat_template(messages, **self.template_kwargs, **kw)

  def to_openai_message(self, message: Message) -> dict:
    converted = super().to_openai_message(message)
    for call in converted.get("tool_calls", []):
      call["function"]["arguments"] = json.loads(call["function"]["arguments"])  # the template reads them as a dict
    # a sampled turn is written from its tokens; the template is only asked what follows its tool calls
    return converted | {"content": ""} if SAMPLED_TOKENS in message else converted

  def build_generation_prompt(self, messages: list[Message], role: str = "assistant", prefill: str | None = None):
    if role != "assistant" or prefill:
      raise NotImplementedError("only a model turn, without a prefill")
    tools = [{"type": "function", "function": t} for t in messages[0].get("tools", [])] if messages else []
    chat = [self.to_openai_message(m) for m in messages]
    sampled_at = [i for i, m in enumerate(messages) if SAMPLED_TOKENS in m]
    tokens = self.render(chat[: sampled_at[0] if sampled_at else None], tools=tools or None, add_generation_prompt=True)
    for i, next_index in zip(sampled_at, [*sampled_at[1:], len(messages)]):
      tokens += messages[i][SAMPLED_TOKENS]
      tokens += self.continuation(chat[i], chat[i + 1 : next_index], tokens[-1])
    return tinker.ModelInput.from_ints(tokens)

  def continuation(self, assistant: dict, replies: list[dict], last: int) -> list[int]:
    """The tokens that follow a sampled turn: the replies to it and the next turn's opening, as the template writes them."""
    alone, with_replies = self.render([assistant]), self.render([assistant, *replies], add_generation_prompt=True)
    if with_replies[: len(alone)] != alone:
      raise ValueError("the template rewrote a turn when its replies were added")
    # sampling stops at the stop token that ends the turn, with or without returning it
    at = max(i for i, t in enumerate(alone) if t in self.stop_tokens)
    return ([] if last == alone[at] else [alone[at]]) + with_replies[at + 1 :]

  def parse_response(self, response: list[int]) -> tuple[Message, ParseTermination]:
    text = self.tokenizer.decode(response, skip_special_tokens=False)
    message: Message = {"role": "assistant", "content": text, SAMPLED_TOKENS: list(response)}
    # after tool replies the template has already opened the thinking channel
    thinking_open = "<channel|>" in text and not text.startswith("<|channel>")
    try:
      prefix = self.prefix + ("<|channel>thought\n" if thinking_open else "")
      parsed = parse_response(text, self.response_template, prefix=prefix)
    except ValueError:
      return message, ParseTermination.MALFORMED
    calls = parsed.get("tool_calls") or []
    if not calls and "<|tool_call>" in text:
      return message, ParseTermination.MALFORMED
    thinking = parsed.get("thinking") or parsed.get("reasoning_content") or ""
    message["content"] = [{"type": "thinking", "thinking": thinking}] if thinking else []
    message["content"].append({"type": "text", "text": parsed.get("content") or ""})
    if calls:
      functions = [c["function"] for c in calls]
      bodies = [ToolCall.FunctionBody(name=f["name"], arguments=json.dumps(f["arguments"])) for f in functions]
      message["tool_calls"] = [ToolCall(id=f"call_{n}", function=body) for n, body in enumerate(bodies)]
    return message, ParseTermination.STOP_SEQUENCE if response[-1] in self.stop_tokens else ParseTermination.EOS
