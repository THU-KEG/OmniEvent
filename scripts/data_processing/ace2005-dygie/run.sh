# pip install -r requirements.txt
# python -m spacy download en_core_web_sm
bash collect_ace_event.sh $1 
python parse_ace_event.py default-settings 
python convert_to_openee.py 