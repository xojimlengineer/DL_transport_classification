import streamlit as st
from fastai.vision.all import*
import plotly.express as px
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath=pathlib.WindowsPath

#title
st.title("Transport klassifikatsiya qiluvchi model ")

#rasmni yuklash
file = st.file_uploader("Rasmni yuklash" , type = ['png','jpeg','gif','svg','jfif'])
if file:
  st.image(file)

  #PIlImage
  img = PILImage.create(file)

  #model
  model = load_learner('transport_clas.pkl')

  #prediction
  pred,pred_id,prob = model.predict(img)
  st.info(f"Ehtimollik: {pred} :{prob[pred_id]*100:.1f}%")
  #plotting
  fig = px.bar(x=prob*100 , y=model.dls.vocab)
  st.plotly_chart(fig)





