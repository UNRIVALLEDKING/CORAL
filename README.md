# Marine & Coral Species Classification App

A unified Streamlit app for classifying marine and coral species using deep learning (PyTorch, Grad-CAM). Supports both marine (ResNet18) and coral (custom CNN) models with Grad-CAM visualizations.

## Features
- Upload an image and classify as marine or coral species
- Grad-CAM heatmap visualization for model interpretability
- Model info, training details, and class descriptions in sidebar

## Setup
1. Clone this repo:
   ```sh
   git clone https://github.com/yourusername/marine-coral-classification.git
   cd marine-coral-classification
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. **Download model weights:**
   - Place your marine model weights in `models/marine_model_retrained_weights.pth`
   - Place your coral model weights in `models/corel_cnn.pth`
   - (Weights are not included due to size. Provide download links here if available.)
4. Run the app:
   ```sh
   streamlit run app.py
   ```

## Deployment (Streamlit Cloud)
- Push this repo to GitHub
- Deploy on [Streamlit Cloud](https://streamlit.io/cloud) by connecting your repo and selecting `app.py` as the main file
- Make sure to add/download model weights after deployment

## Author
Ashutosh Kumar

---
*Model weights are not included in this repository. Please download them separately as described above.* 