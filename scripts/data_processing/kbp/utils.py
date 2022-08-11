from typing import Dict, List, Union


def token_pos_to_char_pos(tokens: List[str],
                          token_pos: List[int]) -> List[int]:
    """Converts the token-level position of a mention into character-level.

    Converts the token-level position of a mention into character-level by counting the number of characters before the
    start position of the mention. The end position could then be derived by adding the character-level start position
    and the length of the mention's span.

    Args:
        tokens (`List[str]`):
            A list of strings representing the tokens within the source text.
        token_pos (`List[int]`):
            A list of integers indicating the word-level start and end position of the mention.

    Returns:
        `List[int]`:
            A list of integers representing the character-level start and end position of the mention.
    """
    word_span = " ".join(tokens[token_pos[0]:token_pos[1]])
    char_start, char_end = -1, -1
    curr_pos = 0
    for i, token in enumerate(tokens):
        if i == token_pos[0]:
            char_start = curr_pos
            break 
        curr_pos += len(token) + 1
    assert char_start != -1
    char_end = char_start + len(word_span) 
    sen = " ".join(tokens)
    assert sen[char_start:char_end] == word_span
    return [char_start, char_end]


def generate_negative_trigger_per_item(item: Dict[str, Union[str, List]]):
    """Generates negative triggers based on the triggers and source text.

    Generates negative triggers based on the triggers and source text. The tokens not within any trigger are regarded
    as negative triggers. The id, trigger word, and word-level position of each negative trigger are stored in a
    dictionary.

    Args:
        item (`Dict`):
            A dictionary containing the annotations of a sentence, including its id, source text, and the event trigger,
            argument, and entity annotations of the sentence.

    Returns:
        item (`Dict`):
            A dictionary similar to the input dictionary but added the negative triggers annotations.
    """
    tokens = item["text"].split()
    trigger_position = {i: False for i in range(len(tokens))}
    for event in item["events"]:
        for trigger in event["triggers"]:
            start_pos = len(item["text"][:trigger["position"][0]].split())
            end_pos = start_pos + len(trigger["trigger_word"].split())
            for pos in range(start_pos, end_pos):
                trigger_position[pos] = True 
    item["negative_triggers"] = []
    for i, token in enumerate(tokens):
        if trigger_position[i] or token == "":
            continue
        _event = {
                "id": len(item["negative_triggers"]),
                "trigger_word": tokens[i],
                "position": token_pos_to_char_pos(tokens, [i, i+1])
        }
        item["negative_triggers"].append(_event)
    return item 


def generate_negative_trigger(data: Dict[str, Union[str, List]],
                              none_event_instances: List[Dict[str, Union[str, List[str]]]]):
    """Generates negative triggers from the none-event instances.

    Generates negative triggers from the none-event instances, in which the tokens of the none-event sentences are
    regarded as negative triggers.

    Args:
        data (`Dict`):
            A dictionary containing the annotations of a sentence, including its id, source text, and the event trigger,
            argument, and entity annotations of the sentence.
        none_event_instances (`Dict`):
            A list of dictionaries containing the sentences that do not contain any event triggers and entities.

    Returns:
        A dictionary similar to the input dictionary but added the negative triggers annotations.
    """
    for item in data:
        tokens = item["text"].split()
        trigger_position = {i: False for i in range(len(tokens))}
        for event in item["events"]:
            for trigger in event["triggers"]:
                start_pos = len(item["text"][:trigger["position"][0]].split())
                end_pos = start_pos + len(trigger["trigger_word"].split())
                for pos in range(start_pos, end_pos):
                    trigger_position[pos] = True 
        item["negative_triggers"] = []
        for i, token in enumerate(tokens):
            if trigger_position[i] or token == "":
                continue
            _event = {
                    "id": len(item["negative_triggers"]),
                    "trigger_word": tokens[i],
                    "position": token_pos_to_char_pos(tokens, [i, i+1])
            }
            item["negative_triggers"].append(_event)
    none_event_data = []
    for ins_idx, item in enumerate(none_event_instances):
        for sentence in item["sentences"]:
            refined_sen_events = dict(id="%s-%d"%(item["id"], len(data)+ins_idx))
            refined_sen_events["text"] = sentence
            refined_sen_events["events"] = []
            refined_sen_events["negative_triggers"] = []
            refined_sen_events["entities"] = []
            tokens = sentence.split()
            for i, token in enumerate(tokens):
                _none_event = {
                    "id": len(refined_sen_events["negative_triggers"]),
                    'trigger_word': tokens[i],
                    'position': token_pos_to_char_pos(tokens, [i, i+1])
                }
                refined_sen_events["negative_triggers"].append(_none_event)
            none_event_data.append(refined_sen_events)
    all_data = data + none_event_data
    return all_data