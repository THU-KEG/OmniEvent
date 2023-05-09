


def output_standardization(
        predictions,
        instances,
        mode="ED",
        event_predictions=None
):
    """
    Args:
        predictions (`List[List[]]`): 
            Original predictions:
            - Sequence labeling. [[O, O, B-Attack, I-Attack, O, ...], ...]
            - MRC. [[(2, 4, Attack), ...], ...]
            - Seq2Seq. [[(<span>, <type>), ...], ...]
        instances (`[List[Dict[]]]`):
            Original processed instances
        mode (`str`):
            "ED" or "EAE".
        event_predictions (`List[str]`):
            The event prediction which is corresponding to each item in predictions (only used for EAE mode)
    
    Returns:
        standard_predcitions (`List[List[str]]`):
            Standardized predictions.
        standard_labels (`List[List[str]]`):
            Standardized labels.
    """
    if mode == "ED":
        return output_standardization_ed(predictions, instances)
    elif mode == "EAE":
        return output_standardization_eae(predictions, instances, event_predictions)
    else:
        raise ValueError(f"Invalid `mode`: {mode}")


def output_standardization_ed(
        predictions,
        instances
):
    pass


def output_standardization_eae(
        predictions,
        instances,
        event_predictions
):
    pass

