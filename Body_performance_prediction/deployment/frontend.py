import streamlit as st
import requests

### Set page 
st.set_page_config(
    page_title="Khalisul_Akbar-Batch 10",
    page_icon="🏋️‍♂️",
    layout="centered",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.google.com',
        'Report a bug': "https://github.com/Khal0000",
        'About': "# This is my first dashboarb!!!"
    }
)

st.markdown("<h1 style='text-align: center; color: white;'>Body Performance Prediction</h1>", unsafe_allow_html=True)
st.write('---------------------------------------------------\n\n\n')
st.sidebar.image("https://cdn.pixabay.com/photo/2016/08/02/02/27/woman-1562560_960_720.jpg", width=300)
st.sidebar.subheader('Physical appearance :')

age = st.sidebar.slider("Age", 0, 130, 0)
gender = st.sidebar.selectbox("Gender", ["M","F"])
height = st.sidebar.number_input("Height in Cm", 0.0,300.0,0.0, step=1.0)
weight = st.sidebar.number_input("Weight in Kg", 0.0,300.0,0.0, step=1.0)
body_fat = st.sidebar.number_input("Body Fat Percentage", 0.0,100.0,0.0, step=1.0)

st.sidebar.write('---------------------------------------------------\n\n\n')
st.sidebar.subheader('Medical Test :')
st.sidebar.write('Tell us your blood pressure result')

diastolic = st.sidebar.number_input("Diastolic in mmHg", 0.0,150.0,80.0, step=1.0)
systolic = st.sidebar.number_input("Systolic in mmHg", 0.0,150.0,120.0, step=1.0)

st.sidebar.write('---------------------------------------------------\n\n\n')
st.sidebar.subheader('Physical Test :')

grip_force = st.sidebar.number_input("Grip Force", 0.0,1000.0,0.0, step=1.0)
sit_and_bend_forward = st.sidebar.number_input("Sit and bend forward in Cm ", 0.0,1000.0,0.0, step=1.0)
sit_ups_counts = st.sidebar.number_input("Sit-ups Counts", 0.0,1000.0,0.0, step=1.0)
broad_jump = st.sidebar.number_input("Broad Jump in Cm", 0.0,1000.0,0.0, step=1.0)
st.sidebar.write('---------------------------------------------------\n\n\n')

# inference
data = {"age" : age ,
        "gender" : gender,
         "height" : height,
         "weight" : weight, 
         "body_fat" : body_fat,
         "diastolic" : diastolic,
         "systolic" : systolic,
          "grip_force" : grip_force,
          "sit_and_bend_forward" : sit_and_bend_forward,
          "sit_ups_counts" : sit_ups_counts,
          "broad_jump" : broad_jump}

URL = "https://h8-test-backend.herokuapp.com/body"

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader('Submitted Data')
    st.write('---------------------------------------------------')
    st.write(data)

    submit = st.button("Predict")
    
with col2:
    # komunikasi
    r = requests.post(URL, json=data)
    res = r.json()
    st.subheader('Prediction Result\n\n\n\n\n\n\n')
    st.write('---------------------------------------------------')
    if res['code'] == 200 and submit:
        # st.subheader('Your Result is :')
        st.markdown(f"<h1 style='text-align: center; color: Gold; font-size: 230px;'>{res['result']['classes']}</h1>", unsafe_allow_html=True)

st.write('---------------------------------------------------\n')
# st.write('Submitted data : \n---------------------------------------------------')
# st.write(data)
# st.write('---------------------------------------------------\n')
# submit = st.button("Predict")

