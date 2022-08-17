import sys 
sys.path.append("..")
from OpenEE.infer import infer

text = "U.S. and British troops were moving on the strategic southern port city of Basra Saturday after a massive aerial assault pounded Baghdad at dawn"
infer(text, task="EE")