import torch
from torchvision import transforms
from PIL import Image
import os

class PredictionPipeline:
    def __init__(self, filename):
        self.filename = filename

    def predict(self):
        # Load model
        model =  torch.load('checkpoints\checkpoint_model.pth')
        model.eval()

        # Preprocess the image
        imagename = self.filename
        test_image = Image.open(imagename).convert('RGB')
        preprocess = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        test_image = preprocess(test_image)
        test_image = test_image.unsqueeze(0)

        # if result == 0:
        #     prediction = 'Tumor'
        # else:
        #     prediction = 'Normal'

        # return [{"image": prediction}]


        def output_label(label):
            output_mapping = {
                        0: "T-shirt/Top",
                        1: "Trouser",
                        2: "Pullover",
                        3: "Dress",
                        4: "Coat",
                        5: "Sandal",
                        6: "Shirt",
                        7: "Sneaker",
                        8: "Bag",
                        9: "Ankle Boot"
                        }
            input = (label.item() if type(label) == torch.Tensor else label)
            return output_mapping[input]
        
                # Make prediction
        with torch.no_grad():
            output = model(test_image)
            result = torch.argmax(output, dim=1).item()
        print(output_label(result))
        
        return output_label(result)