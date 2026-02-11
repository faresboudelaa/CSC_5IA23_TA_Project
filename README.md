# CSC_5IA23_TA_Project
Robust Computer vision with deep learning, XAI, Uncertainty quantification (CSC_5IA23_TA) Project


## Link for the trained network
Resnet trained for 360 epochs

### <a href= "https://drive.google.com/file/d/1tMgoqWBzVRWpsmKJHyY_0p0oAWbqNmI6/view?usp=drive_link"> link </a>

To load the model, use the following code snippet: `load_model(path)`

TO extract features use

```python
resnet = load_model(path)
features = resnet.model(x)

classifier = resnet.cl

logits = resnet(x)
```
