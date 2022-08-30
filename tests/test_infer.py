import json 
import unittest
import sys 
sys.path.append("..")
from OmniEvent.infer import infer

class TestInfer(unittest.TestCase):

    def test_seq2seq(self):
        input_text = "U.S. and British troops were moving on the strategic southern port city of Basra Saturday after a massive aerial assault pounded Baghdad at dawn"
        result = infer(task="EE", text=input_text)
        result = sorted(result[0]["events"], key=lambda item: item["trigger"])
        print(json.dumps(result, indent=4))
        self.assertEqual(result[0]["trigger"], "assault")
        self.assertEqual(result[0]["type"], "attack")
        self.assertEqual(result[1]["trigger"], "pounded")
        self.assertEqual(result[1]["type"], "injure")


if __name__ == "__main__":
    unittest.main()


