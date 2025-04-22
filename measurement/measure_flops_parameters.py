from fvcore.nn import FlopCountAnalysis,flop_count_table
import torch

class Measure_FLOPs_Parameters:
    def __init__(self,model,input_shape):
        self.model=model
        self.input_shape=input_shape
    def __call__(self):
        self.model=self.model.to("cuda:0")
        self.model.eval()
        inputs= torch.rand(1,3, self.input_shape,self.input_shape)
        inputs=inputs.to("cuda")
        flops=FlopCountAnalysis(self.model,inputs)
        print(flop_count_table(flops))