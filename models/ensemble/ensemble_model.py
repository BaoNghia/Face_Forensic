import torch
import torch.nn as nn
import torch.nn.functional as F
from models.efficientnet_transfer.model import Efficientnet_transfer
from models.cnn_spp.restnet_spp import RestnetSPP

class MyEnsemble(nn.Module):
    def __init__(self, modelA, modelB, nb_classes):
        super(MyEnsemble, self).__init__()
        self.modelA = modelA
        self.modelB = modelB
        # Remove last linear layer
        self.modelA.fc = nn.Identity()
        self.modelB.fc = nn.Identity()
        self.classifier = nn.Linear(2048+512, nb_classes)
        
    def forward(self, x):
        # modelA
        xA = self.modelA(x.clone())  # clone to make sure x is not changed by inplace methods
        xA = xA.view(xA.size(0), -1)
        # modelB
        xB = self.modelB(x.clone())
        xB = xB.view(xB.size(0), -1)
        x = torch.cat((xA, xB), dim=1)
        x = self.classifier(F.relu(x))
        return x

if __name__ == "__main__":
    # Create models and load state_dicts
    modelA = Efficientnet_transfer(
                name = "efficientnet-b1",
                num_class = 3,
                img_height= 50,
                img_width= 50
    )
    modelB = RestnetSPP(
        name = "resnet32",
        num_class = 3,
        img_height= 50,
        img_width= 50
    )
    # Load state dicts
    modelA.load_state_dict(torch.load(PATH))
    modelB.load_state_dict(torch.load(PATH))

    ensemble_model = MyEnsemble(modelA, modelB)
    x = torch.randn(1, 10), torch.randn(1, 20)
    output = ensemble_model(x)
