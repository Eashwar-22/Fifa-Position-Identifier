import streamlit as st
import random
import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pickle import load


scaler = load(open('../Models/scaler.pkl', 'rb'))
model = tf.keras.models.load_model('../Models/model_1')

class_map=['Attacking Mid',
           'Centre Back',
           'Central Mid',
           'Defensive Mid',
           'Left Back',
           'Left Mid',
           'Right Back',
           'Right Mid',
           'Striker']
work_rates=['High/High','High/Medium','High/Low',
            'Medium/High','Medium/Medium','Medium/Low',
            'Low/High','Low/Medium','Low/Low']

columns=[
       'attacking_crossing', 'attacking_finishing',
       'attacking_heading_accuracy', 'attacking_short_passing',
       'attacking_volleys', 'skill_dribbling', 'skill_curve',
       'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
       'movement_acceleration', 'movement_sprint_speed', 'movement_agility',
       'movement_reactions', 'movement_balance', 'power_shot_power',
       'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
       'mentality_aggression', 'mentality_interceptions',
       'mentality_positioning', 'mentality_vision', 'mentality_penalties',
       'mentality_composure', 'defending_marking_awareness',
       'defending_standing_tackle', 'defending_sliding_tackle', 'High/Low',
       'High/Medium', 'Low/High', 'Low/Low', 'Low/Medium', 'Medium/High',
       'Medium/Low', 'Medium/Medium']
stats={}



st.set_page_config(layout="wide")

st.title("Player Position Recommendation")
st.header("FIFA 22 Database")
st.markdown("---")


st.markdown(''' ## Work Rate ''')
work_rate = st.selectbox("Work Rate",work_rates)



col1,col2,col3=st.columns(3)
with col1:
       st.markdown(''' ## Attacking ''')
       stats['attacking_crossing'] = st.slider('Crossing', 10, 99, 50)
       stats['attacking_finishing'] = st.slider('Finishing', 10, 99, 50)
       stats['attacking_heading_accuracy'] = st.slider('Heading', 10, 99, 50)
       stats['attacking_short_passing'] = st.slider('Short Passing', 10, 99, 50)
       stats['attacking_volleys'] = st.slider('Volleys', 10, 99, 50)
with col2:
       st.markdown(''' 
       ## Skills ''')
       stats['skill_dribbling'] = st.slider('Dribbling', 10, 99, 50)
       stats['skill_curve'] = st.slider('Curve', 10, 99, 50)
       stats['skill_fk_accuracy'] = st.slider('Free Kick Accuracy', 10, 99, 50)
       stats['skill_long_passing'] = st.slider('Long Passing', 10, 99, 50)
       stats['skill_ball_control'] = st.slider('Ball Control', 10, 99, 50)
with col3:
       st.markdown(''' 
       ## Pace ''')
       stats['movement_acceleration'] = st.slider('Acceleration', 10, 99, 50)
       stats['movement_sprint_speed'] = st.slider('Sprint Speed', 10, 99, 50)
       stats['movement_sprint_agility'] = st.slider('Agility', 10, 99, 50)
       stats['movement_reactions'] = st.slider('Reactions', 10, 99, 50)
       stats['movement_balance'] = st.slider('Balance', 10, 99, 50)

col4,col5,col6 = st.columns(3)
with col4:
       st.markdown(''' 
       ## Mentality ''')
       stats['mentality_aggression'] = st.slider('Aggression', 10, 99, 50)
       stats['mentality_interceptions'] = st.slider('Interceptions', 10, 99, 50)
       stats['mentality_positioning'] = st.slider('Positioning', 10, 99, 50)
       stats['mentality_vision'] = st.slider('Vision', 10, 99, 50)
       stats['mentality_penalties'] = st.slider('Penalties', 10, 99, 50)
       stats['mentality_composure'] = st.slider('Composure', 10, 99, 50)
with col5:
       st.markdown(''' 
       ## Power ''')
       stats['power_shot_power'] = st.slider('Shot Power', 10, 99, 50)
       stats['power_jumping'] = st.slider('Jumping', 10, 99, 50)
       stats['power_stamina'] = st.slider('Stamina', 10, 99, 50)
       stats['power_strength'] = st.slider('Strength', 10, 99, 50)
       stats['power_long_shots'] = st.slider('Long Shots', 10, 99, 50)
with col6:
       st.markdown(''' 
       ## Defending ''')
       stats['defending_marking_awareness'] = st.slider('Marking Awareness', 10, 99, 50)
       stats['defending_standing_tackle'] = st.slider('Stand Tackle', 10, 99, 50)
       stats['defending_sliding_tackle'] = st.slider('Slide Tackle', 10, 99, 50)


stats_list=list(stats.values())

if work_rate=='High/High':
       stats_list = stats_list + [0, 0, 0, 0, 0, 0, 0, 0]
elif work_rate=='High/Low':
       stats_list = stats_list + [1, 0, 0, 0, 0, 0, 0, 0]
elif work_rate=='High/Medium':
       stats_list = stats_list + [0, 1, 0, 0, 0, 0, 0, 0]
elif work_rate=='Low/High':
       stats_list = stats_list + [0, 0, 1, 0, 0, 0, 0, 0]
elif work_rate=='Low/Low':
       stats_list = stats_list + [0, 0, 0, 1, 0, 0, 0, 0]
elif work_rate=='Low/Medium':
       stats_list = stats_list + [0, 0, 0, 0, 1, 0, 0, 0]
elif work_rate=='Medium/High':
       stats_list = stats_list + [0, 0, 0, 0, 0, 1, 0, 0]
elif work_rate=='Medium/Low':
       stats_list = stats_list + [0, 0, 0, 0, 0, 0, 1, 0]
elif work_rate=='Medium/Medium':
       stats_list = stats_list + [0, 0, 0, 0, 0, 0, 0, 1]



X= tf.expand_dims(np.array(stats_list),1)
X= tf.reshape(X,[1,37])

# scaler = MinMaxScaler()
X=scaler.transform(X)
st.sidebar.write(X)


prediction_prob=model.predict(X)
prediction=class_map[np.argmax(prediction_prob)]



st.sidebar.markdown('''
---
## Recommended player position gets updated here  ''')
def _update_text_box(value):
    st.session_state["output_value"] = value
if "output_value" not in st.session_state:
    st.session_state["output_value"] = ''


st.sidebar.text_input('Position',
                      value=prediction,
                      key='output_value')
# st.sidebar.button("Check Position",
#                   on_click=_update_text_box,
#                   kwargs={"value": prediction })






