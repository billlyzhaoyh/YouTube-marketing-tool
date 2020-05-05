import streamlit as st
import pandas as pd
import numpy as np
import pickle
from PIL import Image
import tensorflow as tf

from gpt2 import load_model, generate
import ktrain
from automl import load_vision

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

@st.cache(allow_output_mutation=True)
def loader_automl():
    return load_vision()

st.title('YouTube Marketing Tool Demo')


st.write("""
    here we help you better manage your video tags to boost your video's engagement and discoverability
    """)


if st.checkbox('Generate description and tags from title with the help of our AI writer'):

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

if st.checkbox('Genearte tags from thumbnail image'):

	uploaded_img = st.file_uploader("Alternatively choose a thumbnail image to upload", type=['png', 'jpg'])
	if uploaded_img is not None:
		interpreter,input_details,output_details=loader_automl()
		category_tag=loader_categorytag()
		img = np.array(Image.open(uploaded_img)) 
		input_shape = input_details[0]['shape']
		input_data=tf.image.resize(img, [224,224])
		# tf.dtypes.cast(input_data, tf.uint8)
		input_data=tf.cast(input_data, dtype=tf.uint8)
		input_data=tf.expand_dims(input_data, 0) 
		interpreter.set_tensor(input_details[0]['index'], input_data)
		interpreter.invoke()
		# The function `get_tensor()` returns a copy of the tensor data.
		# Use `tensor()` in order to get a pointer to the tensor.
		output_data = interpreter.get_tensor(output_details[0]['index'])
		category=int(tf.math.argmax(output_data,1))
		id_map={0:'Entertainment',1:'News & Politics',2:'People & Blogs',3:'Music',4:'Sports',5:'Comedy'}
		st.write(category_tag[id_map[category]])