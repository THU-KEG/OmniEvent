# Evaluation Module

## Background
Event Extraction (EE) is a challenging task and has captured huge attention from the community. However, there exists two main issues regarding the evaluation process: 
- **Uncomparable Metrics Across Paradigms**

	Various **Event Extraction (EE)** paradigms have been proposed, such **Token Classification**, **Sequence Labeling**, **Seq2Seq** and **MRC**. The implementations of these methods are different and their metrics are paradigm-dependent. Therefore, it is ***not fair*** to directly compare the evaluation results of different paradigms. 

- **Inconsisent Numbers of EAE Instances**

	Previous works tend to break Event Extraction into two subtasks: **Event Detection (ED)** and **Event Argument Extraction (EAE)**. When evaluating the performance of EAE, some people use the gold triggers as the input, while others choose the predicted triggers produced by the ED models. Moreover, there is no standard convention of how to deal with the incorrectly predicted triggers, which leads to inconsitent numbers of instances when evaluating the EAE performance.

## Unified Evaluation
OmniEvent provides a unified evaluation process to tackle the issues above.

- Convert the predictions of different paradigms to word level
	
    The predictions of different paradigms are unifiedly converted to word level. In another word, we align the predictions to each word and compute the evaluation metrics in the Token Classification style. Implementations of the conversion function can be found [here](https://github.com/THU-KEG/OmniEvent/blob/50526ea0ba6aa86a885752b4cc2fa8389c8cc11e/OmniEvent/evaluation/convert_format.py#L19-L105)

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

  OmniEvent provides four standard evaluation modes for EAE according to the way of choosing EAE instances.

    - Gold Mode 
      - Use the gold triggers for EAE.
    - Default Mode 
      - Use the predicted triggers for EAE.
      - Skip the gold Non-NA triggers that are predicted as NA.
      - Include the gold NA triggers that are predicted as Non-NA
    - Loose Mode
      - Use the predicted triggers for EAE.
      - Skip the gold Non-NA triggers that are predicted as NA.
      - Skip all gold NA triggers.
    - Strict Mode
      - Use the predicted triggers for EAE.
      - Include all gold Non-NA triggers.
      - Include all gold NA triggers that are predicted as Non-NA.