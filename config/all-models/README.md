# Experiment Configurations on All Models

## Experiment Settings
We run all the implementations of the models on `ACE-EN+`.

The models include CNN, LSTM, CRF, BERT, RoBERTa, T5, etc.


## File Structure

```angular2html
├── eae                         // Event Argument Extraction
│   ├── mrc                        // MRC Paradigm
│   │   ├── bert-base
│   │   │   └── mrc.yaml
│   │   ├── lstm
│   │   │   └── mrc.yaml
│   │   ├── roberta-base
│   │   │   └── mrc.yaml
│   │   └── roberta-large
│   │       └── mrc.yaml
│   ├── s2s                       // Seq2Seq Paradigm
│   │   ├── t5-base
│   │   │   └── t5-base.yaml
│   │   └── t5-large
│   │       └── t5-large.yaml
│   ├── sl                        // Sequence Labeling
│   │   ├── bert-base
│   │   │   ├── crf.yaml
│   │   │   └── wo-crf.yaml
│   │   ├── lstm
│   │   │   ├── crf.yaml
│   │   │   └── wo-crf.yaml
│   │   ├── roberta-base
│   │   │   ├── crf.yaml
│   │   │   └── wo-crf.yaml
│   │   └── roberta-large
│   │       ├── crf.yaml
│   │       └── wo-crf.yaml
│   └── tc                        // Token Classification
│       ├── bert-base
│       │   ├── cls.yaml
│       │   ├── dm.yaml
│       │   ├── marker.yaml
│       │   └── max.yaml
│       ├── cnn
│       │   ├── dm.yaml
│       │   ├── marker.yaml
│       │   └── max.yaml
│       ├── roberta-base
│       │   ├── cls.yaml
│       │   ├── dm.yaml
│       │   ├── marker.yaml
│       │   └── max.yaml
│       └── roberta-large
│           ├── cls.yaml
│           ├── dm.yaml
│           ├── marker.yaml
│           └── max.yaml
└── ed                         // Event Detection
    ├── s2s                       // Seq2Seq Paradigm
    │   ├── mt5-base
    │   │   └── mt5-base.yaml
    │   ├── mt5-large
    │   │   └── mt5-large.yaml
    │   ├── t5-base
    │   │   └── t5-base.yaml
    │   └── t5-large
    │       └── t5-large.yaml
    ├── sl                        // Sequence Labeling Paradigm
    │   ├── bert-base
    │   │   ├── crf.yaml
    │   │   └── wo-crf.yaml
    │   ├── lstm
    │   │   ├── crf.yaml
    │   │   └── wo-crf.yaml
    │   ├── roberta-base
    │   │   ├── crf.yaml
    │   │   └── wo-crf.yaml
    │   └── roberta-large
    │       ├── crf.yaml
    │       └── wo-crf.yaml
    └── tc                        // Token Classification Paradigm
        ├── bert-base
        │   ├── cls.yaml
        │   ├── dm.yaml
        │   ├── marker.yaml
        │   └── max.yaml
        ├── cnn
        │   ├── dm.yaml
        │   ├── marker.yaml
        │   └── max.yaml
        ├── roberta-base
        │   ├── cls.yaml
        │   ├── dm.yaml
        │   ├── marker.yaml
        │   └── max.yaml
        └── roberta-large
            ├── cls.yaml
            ├── dm.yaml
            ├── marker.yaml
            └── max.yaml
```