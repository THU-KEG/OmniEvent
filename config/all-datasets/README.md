# Experiment Configurations on All Datasets

## Experiment Settings
For all Chinese datasets, we adopt `BERT-base-chinese` and `mT5-base` as the backbone.

For all English datasets, we adopt `BERT-base-uncased` and `T5-base` as the backbones.

For Token Classification paradigm, we use the `marker` aggregation. 

For Sequence Labeling paradigm, we use the `linear` head.


## File Structure

```angular2html
├── eae                      //Event Argument Extraction 
│   ├── mrc                         // MRC Paradigm
│   │   ├── ace-dygie.yaml
│   │   ├── ace-en.yaml
│   │   ├── ace-zh.yaml
│   │   ├── fewfc.yaml
│   │   └── richere.yaml
│   ├── s2s                         // Seq2Seq Paradigm
│   │   ├── ace-dygie.yaml
│   │   ├── ace-en.yaml
│   │   ├── ace-zh.yaml
│   │   ├── dt.yaml
│   │   ├── duee.yaml
│   │   ├── fewfc.yaml
│   │   └── richere.yaml
│   ├── sl                         // Sequence Labeling Paradigm
│   │   ├── ace-dygie.yaml
│   │   ├── ace-en.yaml
│   │   ├── ace-zh.yaml
│   │   ├── duee.yaml
│   │   ├── fewfc.yaml
│   │   └── richere.yaml
│   └── tc                         // Token Classification Paradigm
│       ├── ace-dygie.yaml
│       ├── ace-en.yaml
│       ├── ace-zh.yaml
│       ├── fewfc.yaml
│       ├── kbp.yaml
│       └── richere.yaml
└── ed                      // Event Detection
    ├── s2s                       // Seq2Seq Paradigm
    │   ├── ace-dygie.yaml
    │   ├── ace-en.yaml
    │   ├── ace-zh.yaml
    │   ├── dt.yaml
    │   ├── duee.yaml
    │   ├── fewfc.yaml
    │   ├── leven.yaml
    │   ├── maven.yaml
    │   └── richere.yaml
    ├── sl                        // Sequence Labeling Paradigm
    │   ├── ace-dygie.yaml
    │   ├── ace-en.yaml
    │   ├── ace-zh.yaml
    │   ├── duee.yaml
    │   ├── fewfc.yaml
    │   ├── leven.yaml
    │   ├── maven.yaml
    │   └── richere.yaml
    └── tc                        // Token Classification Paradigm
        ├── ace-dygie.yaml
        ├── ace-en.yaml
        ├── ace-zh.yaml
        ├── duee.yaml
        ├── fewfc.yaml
        ├── kbp.yaml
        ├── leven.yaml
        ├── maven.yaml
        └── richere.yaml
```
