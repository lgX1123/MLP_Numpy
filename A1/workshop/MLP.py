class MLP(object):
    def __init__(self):
        self.layers = []
        self.params = []
        self.num_layers = 0
    
    def add_layer(self, layer):
        self.layers.append(layer)
        if layer.requires_grad:
            if hasattr(layer, 'W'):
                self.params.append(layer.W)
            if hasattr(layer, 'b'):
                self.params.append(layer.b)
            if hasattr(layer, 'gamma'):
                self.params.append(layer.gamma)
            if hasattr(layer, 'beta'):
                self.params.append(layer.beta)
        self.num_layers += 1

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, x):
        for layer in self.layers[::-1]:
            x = layer.backward(x)
        return x