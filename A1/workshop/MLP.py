class MLP(object):
    def __init__(self):
        self.layers = []
        self.params = []
        self.num_layers = 0
    
    def add_layer(self, layer):
        self.layers.append(layer)
        if layer.requires_grad:
            self.params.append(layer.W)
            self.params.append(layer.b)
        self.num_layers += 1

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x):
        for layer in self.layers[::-1]:
            x = layer.backward(x)
        return x