from colorama import Fore, Style
from enum import Enum
from typing import Generic, TypeVar
import torch
import json

def print_dict(some_dict):
    for key in some_dict:
        print(key, some_dict[key])

def print_section():
    print("*" * 20)

def print_green(some_string):
    print(Fore.GREEN)
    print(some_string)
    print(Style.RESET_ALL)

def colorize_string(color: str, string) -> str:
    return color + string + Style.RESET_ALL

def red(obj_to_color) -> str:
    return colorize_string(Fore.RED, str(obj_to_color))

def green(obj_to_color) -> str:
    return colorize_string(Fore.GREEN, str(obj_to_color))

def blue(obj_to_color) -> str:
    return colorize_string(Fore.BLUE, str(obj_to_color))

def magenta(obj_to_color) -> str:
    return colorize_string(Fore.MAGENTA, str(obj_to_color))

def unsupported_type_error(x):
    return RuntimeError("Unhandled type: {}".format(type(x).__name__))

def die(message):
    raise RuntimeError(message)

def tensor_shape(tensor: torch.Tensor):
    return list(tensor.shape)

class OptionState(Enum):
    Something = 1
    Nothing = 2


T = TypeVar('T')


class Option(Generic[T]):
    def __init__(self, val: T):
        if val is None:
            self.state = OptionState.Nothing
        else:
            self.state = OptionState.Something
            self.value = val

    def get_value(self) -> T:
        if self.state == OptionState.Nothing:
            raise RuntimeError("Trying to access nothing")
        return self.value

    def is_nothing(self) -> bool:
        return self.state == OptionState.Nothing

    def is_something(self) -> bool:
        return self.state == OptionState.Something


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device", device)


def pretty_string(obj) -> str:
    return json.dumps(obj=obj, indent=4)
