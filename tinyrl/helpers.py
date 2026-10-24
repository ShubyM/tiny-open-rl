import os, functools
from typing import Any, TypeVar, overload
T = TypeVar("T")

# NOTE: helpers is not allowed to import from anything else in tinyrl
@overload
def getenv(key:str) -> int: ...
@overload
def getenv(key:str, default:T) -> T: ...
@functools.cache
def getenv(key:str, default:Any=0): return type(default)(os.getenv(key, default))

DEBUG = getenv("DEBUG")
