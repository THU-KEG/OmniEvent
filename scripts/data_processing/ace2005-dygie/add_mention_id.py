import jsonlines
import os


if __name__ == "__main__":
    input_path = "../../../data/processed/ace2005-dygie"

    for file in os.listdir(input_path):
        if "unified" in file:
            data = list(jsonlines.open(os.path.join(input_path, file)))
            with jsonlines.open(os.path.join(input_path, file), "w") as f:
                for d in data:
                    for event in d["events"]:
                        for trigger in event["triggers"]:
                            for argument in trigger["arguments"]:
                                for mention in argument["mentions"]:
                                    left_pos, right_pos = mention["position"]
                                    mention["mention_id"] = '{}-{}-{}'.format(mention['mention'], left_pos, right_pos)
                    jsonlines.Writer.write(f, d)
