import os 
from pydantic import BaseModel 
from typing import List, Union, Tuple


class Input(BaseModel):
    """Input format.
    Args:
        text: User input. 
        language: Selected in `{"English", "Chinese"}`.
        task: Selected in `{"Event Detection", "Event Argument Extraction", "Event Extraction"}`.
        triggers: List of tuple `(trigger word, char offset start in text, char offset end in text)`; 
                    The char offset is left closed right open.
        ontology: Selected in `{"ERE", "MAVEN", "LEVEN", "DuEE", "FewFC"}`
    """
    text: str 
    language: str = "English"
    task: str = "Event Detection"
    ontology: str = "ERE"
    triggers: List[Tuple]= []


class Argument:
    """Argument format.

    Args:
        mention: Mention of the argument in text.
        role: Role type of the argument mention.
        offset (`[int, int]`): Char offset of the mention in the text. The char offset is left closed right open.
    """
    mention: str 
    role: str
    offset: List[int]


class Event:
    """Event format.

    Args:
        trigger: Extracted trigger word of the event.
        type: The predicted event type.
        offset: (`[int, int]`): Char offset ot the trigger.
        arguments: Arguments of the trigger. Please see class `Argument`.
    """
    trigger: str
    type: str 
    offset: List[int]
    arguments: List[Argument] = []


class Result:
    """Return format.

    Args:
        text: The user input.
        events: Extracted events. Please see class `Event`.
    """
    text: str
    events: List[Event] = []


if __name__ == "__main__":
    result = Result()
    result.text = "My name is penghao."
    result.events.append({"name": 123})
    import pdb; pdb.set_trace()