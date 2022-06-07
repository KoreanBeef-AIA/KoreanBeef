from torch import nn

F = nn.functional

class CustomNet(nn.Module):

    def __init__(self, n_inputs:int, n_outputs:int, **kwargs):
        """
        Args:
            n_input(int): feature 수
            n_output(int): class 수

        Notes:
            fc: fully connected layer
        """
        super(CustomNet, self).__init__()
        self.n_input = n_inputs
        self.n_output = n_outputs

        self.linear = nn.Linear(self.n_input, self.n_output)

    def forward(self, x):
        output = self.linear(x)
        
        return output
