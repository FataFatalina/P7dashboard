import streamlit as st
import pandas as pd
import lime
import requests
import json
from types import SimpleNamespace


# Set title
st.title('Credit loan attribution prediction')
st.header("Adèle Souleymanova - Data Science project 7- Openclassrooms")

# Display the LOGO

# files = os.listdir('Image_logo')
# for file in files:
img = Image.open("LOGO.png")
st.sidebar.image(img, width=250)