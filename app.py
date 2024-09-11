import streamlit as st
import os, torch
from src.image_transformer import image_transform
from src.cnn_architecture import XrayCNN



current_dir = os.getcwd()
image_folder = os.path.join(current_dir,"image")
artifacts_folder = os.path.join(current_dir,"artifacts")

def checkimagefolder():
 if os.listdir(image_folder):
  for filename in os.listdir(image_folder):
 
   os.remove(os.path.join(image_folder,filename))

st.title('Chest X-Ray Covid Detection')

# st.subheader('Upload your Chest X-Ray Image here')

image_file = st.file_uploader("Upload Chest X-Ray Image",type=['jpg','png','jpeg'],accept_multiple_files=False)

st.markdown('***Note: This is just a demo app, so please upload x-ray images only***')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = XrayCNN(num_classes=2)
model.to(device)
model.load_state_dict(torch.load(os.path.join(artifacts_folder,'best_model_epoch_8.pth'),map_location=device))
model.eval()
threshold = 0.7
pred_dict = {
  0:'Normal',
  1:'Pnemonia'
}



if st.button('Submit',type="primary"):
 if image_file:
  if (image_file.size > 2*1024*1024):
    st.error("File size greater than 2 MB")
  else:
   checkimagefolder()
   file_storage = os.path.join(image_folder,image_file.name) 
   with open(file_storage,'wb') as f:
     f.write(image_file.read())
   
   
   image = image_transform(file_storage)
   img = image.image_transformer()
   with torch.no_grad():
         output = model(img)


         probabilities = torch.softmax(output, dim=1)
         predicted_class = (probabilities[:, 1] >= threshold).long()
         pred = predicted_class.item()
         st.image(file_storage, caption=image_file.name)
         if(pred==0):
            st.header(f"Patient is Normal")
         elif (pred==1):
            st.header("Patient may have Pneumonia; please consult doctor for further diagnosis and treatment")  
 



