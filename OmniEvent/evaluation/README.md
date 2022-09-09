# Evaluation Module

## Background
Event Extraction (EE) is a challenging task and has captured broad attention from the community. However, there are two main issues regarding the evaluation process: 

- **Uncomparable Metrics Across Paradigms**

	Various Event Extraction (EE) models follow different **paradigms**, such as **Token Classification**, **Sequence Labeling**, **Seq2Seq** and **MRC**. The previous evaluation metrics are often paradigm-dependent. For example, in EAE methods of the Token Classification paradigm, we often choose the entities as the candidates for event arguments, but methods of the Seq2Seq paradigm may not respect the entity boundary. Therefore, it is actually ***not fair*** to directly compare the evaluation results of different paradigms.

- **Inconsisent scope of EAE Evaluation**

	Previous works tend to break Event Extraction into two subtasks: **Event Detection (ED)** and **Event Argument Extraction (EAE)**. When evaluating the performance of EAE, some people use the gold triggers as the input, while others may choose the predicted triggers produced by ED models. Moreover, there is no standard convention of how to deal with the incorrectly predicted triggers, which leads to the inconsitent scope of instances are considered when evaluating EAE.

## Unified Evaluation
OmniEvent provides a unified evaluation process to tackle the issues above.

- Convert the predictions of different paradigms to a unified candidate set
	
    The predictions of different paradigms are converted to predictions on a unified candidate set. In another word, we align the predictions to the candidates of the Token Classification paradigm (words for ED, entities for EAE if the entities are annotated) and compute all the evaluation metrics in the Token Classification style. Implementations of the conversion function can be found [here](https://github.com/THU-KEG/OmniEvent/blob/main/OmniEvent/evaluation/convert_format.py).

    - Convert Sequence Labeling to Token Classification
      <div align='center'>
      <img src="../../imgs/convert-sl.jpeg" style=""></div>
    - Convert Seq2Seq to Token Classification
      <div align='center'>
      <img src="../../imgs/convert-s2s.jpeg" style=""></div>
    - Convert MRC to Token Classification
      <div align='center'>
      <img src="../../imgs/convert-mrc.jpeg" style=""></div>
    
- Provide four standard EAE evaluation modes

  OmniEvent provides four evaluation modes for EAE according to the way of choosing EAE instances, which are used in different previous works.

    - Gold Mode 
      - Use the gold triggers for EAE.
    - Default Mode 
      - Use the predicted triggers for EAE.
      - Skip the gold positive (non-NA) triggers that are predicted as negative (NA).
      - Include the gold negative triggers that are predicted as positive
    - Loose Mode
      - Use the predicted triggers for EAE.
      - Skip the gold positive triggers that are predicted as negative.
      - Skip all gold negative triggers.
    - Strict Mode
      - Use the predicted triggers for EAE.
      - Include all gold positive triggers.
      - Include all gold negative triggers that are predicted as positive.