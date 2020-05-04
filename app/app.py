import streamlit as st
import pandas as pd
import numpy as np
import pickle

from gpt2 import load_model, generate
import ktrain

@st.cache(allow_output_mutation=True)
def loader_gpt2():
    return load_model()

@st.cache(allow_output_mutation=True)
def loader_distilbert():
    return ktrain.load_predictor('./models/distillbert/category_distilbert_predictor')

@st.cache(allow_output_mutation=True)
def loader_categorytag():
	with open('category_tag.p','rb') as f:
		category_tag = pickle.load(f)
	return category_tag

st.title('YouTube Marketing Tool Demo')


st.write("""
    here we help you better manage your video tags to boost your video's engagement and discoverability
    """)

st.header('Try out our AI companion writer for generating description recommendations')
st.subheader('DISCLAIMER: Neural Language Generation feature form GPT2 is an experimental feature')



title = st.text_input("Input your YouTube Video Title to start your experience with us",'Eminem - Walk On Water (Audio) ft. Beyoncé')

if st.checkbox('Want to get inspiration for descriptions from our AI description writer (GPT2)?'):
	model, tokenizer = loader_gpt2()
	max_length = st.slider(
	        """ Max description Length 
	        (Longer length, slower generation)""",
	        50,
	        500,
	        150
	    )
	if st.button("Generate descriptions"):
		if title:
			sample = generate(model,tokenizer,input_text=title,max_length=max_length)
			st.write(sample)
		else:
			st.write('Please input a valid Youtube Video Title')

st.header('Input your descriptions below and we will recommend tags for you to add to your video')
st.subheader('DISCLAIMER: we only support video of category Entertainment ,News & Politics, People & Blogs, Music, Sports and Comedy at the moment due to data avalability')

desciprtion = st.text_area("Input your YouTube Video Description here to get recommended tags",'Eminem\'s new track Walk on Water ft. Beyoncé is available everywhere: http://shady.sr/WOWEminem Playlist Best of Eminem: https://goo.gl/AquNpo Subscribe for more: https://goo.gl/DxCrDV\n\nFor more visit: \nhttp://eminem.com\nhttp://facebook.com/eminem\nhttp://twitter.com/eminem\nhttp://instagram.com/eminem\nhttp://eminem.tumblr.com\nhttp://shadyrecords.com\nhttp://facebook.com/shadyrecords\nhttp://twitter.com/shadyrecords\nhttp://instagram.com/shadyrecords\nhttp://trustshady.tumblr.com\n\nMusic video by Eminem performing Walk On Water. (C) 2017 Aftermath Records\nhttp://vevo.ly/gA7xKt')

if st.button("Generate tags"):
	predictor=loader_distilbert()
	category_tag=loader_categorytag()
	if desciprtion:
		results=predictor.predict(desciprtion)
		st.write(category_tag[results])
	else:
		st.write('Please input a valid Youtube Video Description')