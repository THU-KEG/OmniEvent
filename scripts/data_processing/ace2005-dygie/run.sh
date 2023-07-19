# # Install requirements
# pip install -r requirements.txt
# # Download scpay models
# # We use en_core_web_sm==2.0.0 here: https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.0.0/en_core_web_sm-2.0.0.tar.gz
# python -m spacy download en_core_web_sm
bash collect_ace_event.sh ../../../data/original/ace_2005_td_v7
python parse_ace_event.py default-settings 
python convert_to_openee.py