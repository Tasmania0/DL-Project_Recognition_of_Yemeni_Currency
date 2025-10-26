# DL_Project_Recognition_of_Yemeni_Currency 

##  Project Overview
- Deep learning project using transfer learning from RES18 to make computer vision project for recognition  of Yemeni Currency through images 
- There is the trainig code from google colab where i used transfer learning from res18 and from code you can make your own model.
- To test the model the app.py is responsible for making local web page which has the code index.html
- In the web page you can upload a phot to test (there are 3 photos added in the file).
-Or you can use the Camera to real time recognize Currency.
---

##  Features
- Transfer learning using ResNet18
- Works with front and back sides of Yemeni banknotes
- Pretrained model included for direct testing
- Local Streamlit web interface for testing images
- Modular code for easy retraining on custom data

---

##  Model Details
- Architecture: ResNet18 (pretrained on ImageNet)  
- Framework: PyTorch  
- Input size: 224×224 RGB  
- Optimizer: Adam  
- Loss function: CrossEntropyLoss  
- Data augmentation: Random rotation, flip, and normalization  

---

##  How to Use

### Option 1 — Use the Existing Model
You can directly use the pretrained model provided in this repository:  
Yemeni_currency_classifier.pth  
Run the web interface to test it on sample images.

### Option 2 — Train Your Own Model
If you have your own dataset:
1. Open the notebook yemeni_code_training.ipynb.
2. Replace the dataset path with your data location.
3. Adjust the number of classes and training parameters as needed.
4. Run the notebook to generate a new .pth model file.

*Note:* The dataset used in development is not included in this repository.  
You must prepare your own labeled dataset with separate folders for each currency denomination.

---

##  Testing the Model
To test the model locally using the web interface:

python app.py