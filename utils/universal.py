from colorama import Fore, Style
from enum import Enum
from typing import Generic, TypeVar
import torch
import json
from utils.structs import Annotation
import pandas as pd


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


def read_predictions_file(predictions_file_path) -> dict[str, list[Annotation]]:
    df = pd.read_csv(predictions_file_path, sep='\t')
    sample_to_annos = {}
    for _, row in df.iterrows():
        annos_list = sample_to_annos.get(str(row['sample_id']), [])
        annos_list.append(
            Annotation(
                begin_offset=int(row['begin']),
                end_offset=int(row['end']),
                label_type=str(row['type']),
                extraction=str(row['extraction']),
            )
        )
        sample_to_annos[str(row['sample_id'])] = annos_list
    return sample_to_annos


def f1(TP, FP, FN) -> tuple[float, float, float]:
    """
    Given true-positives, false-positives, and false-negatives
    Returns the f1 score, precision, and recall
    """
    if (TP + FP) == 0:
        precision = None
    else:
        precision = TP / (TP + FP)
    if (FN + TP) == 0:
        recall = None
    else:
        recall = TP / (FN + TP)
    if (precision is None) or (recall is None) or ((precision + recall) == 0):
        return 0, 0, 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)
        return f1_score, precision, recall
