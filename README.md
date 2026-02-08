# CSC_5IA23_TA_Project
Robust Computer vision with deep learning, XAI, Uncertainty quantification (CSC_5IA23_TA) Project


## Link for the trained network
Resnet trained for 360 epochs

### <a href= "https://drive.google.com/file/d/1tMgoqWBzVRWpsmKJHyY_0p0oAWbqNmI6/view?usp=drive_link"> link </a>


```python
#load the model
resnet = ResNet18(64,2,100).to(device)
resnet.load_state_dict(torch.load("resnet_360_epoch.pth"))

```
