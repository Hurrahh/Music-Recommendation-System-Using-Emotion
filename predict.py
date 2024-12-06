import pickle
from io import BytesIO
import librosa
import numpy as np
import csv
import pandas as pd
import tensorflow as tf
import warnings
import sys
from pydub import AudioSegment
import streamlit as st
import tempfile
if not sys.warnoptions:
    warnings.simplefilter("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
model = tf.keras.models.load_model('Emotion_Prediction.keras')

# json_file = open('CNN_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = tf.keras.models.model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("best_model1_weights.keras")
# print("Loaded model from disk")

with open('scaler2.pickle', 'rb') as f:
    scaler2 = pickle.load(f)

with open('encoder2.pickle', 'rb') as f:
    encoder2 = pickle.load(f)

print("Done")


def zcr(data, frame_length, hop_length):
    zcr = librosa.feature.zero_crossing_rate(data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(zcr)


def rmse(data, frame_length=2048, hop_length=512):
    rmse = librosa.feature.rms(y=data, frame_length=frame_length, hop_length=hop_length)
    return np.squeeze(rmse)


def mfcc(data, sr, frame_length=2048, hop_length=512, flatten: bool = True):
    mfcc = librosa.feature.mfcc(y=data, sr=sr)
    return np.squeeze(mfcc.T) if not flatten else np.ravel(mfcc.T)


def extract_features(data, sr=22050, frame_length=2048, hop_length=512):
    result = np.array([])

    result = np.hstack((result,
                        zcr(data, frame_length, hop_length),
                        rmse(data, frame_length, hop_length),
                        mfcc(data, sr, frame_length, hop_length)
                        ))
    return result


def get_predict_feat(path):
    d, s_rate = librosa.load(path, duration=2.5, offset=0.6)
    res = extract_features(d)
    result = np.array(res)
    result = np.reshape(result, newshape=(1, 2376))
    i_result = scaler2.transform(result)
    final_result = np.expand_dims(i_result, axis=2)

    return final_result

emotions1={1:'Neutral', 2:'Calm', 3:'Happy', 4:'Sad', 5:'Angry', 6:'Fear', 7:'Disgust',8:'Surprise'}
def prediction(path1):
    res=get_predict_feat(path1)
    predictions=model.predict(res)
    y_pred = encoder2.inverse_transform(predictions)
    return y_pred[0][0]

def recommendations(s):
    filename = 'songs.csv'
    emotion = ""
    fields = []
    rows = []
    songs = []
    songs1=[]
    if s == 'happy':
        emotion ='Happy'
    elif s == 'sad':
        emotion ='Sad'
    elif s == 'neutral':
        emotion ='Neutral'
    elif s == 'calm':
        emotion ='Calm'
    elif s == 'angry':
        emotion ='Angry'
    else:
        print("no emotion")
    final=[]
    ind_pos = [1,2]


    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            rows.append(row)

        for i in range(csvreader.line_num-1):
            words = rows[i][3].split(",")
            for word in words:
                if word.strip() == emotion:
                    songs.append(rows[i])
            # get total number of rows
        pd.set_option('display.max_rows',500)
        pd.set_option('display.max_columns', 500)
        pd.set_option('display.width', 1000)
        for j in range(len(songs)):
            final.append([songs[j][i] for i in ind_pos])
        pd_dataframe = pd.DataFrame(final, columns=['Artist', 'Song'])
        st.write(" The songs that match your mood are : ")
        return pd_dataframe


def convert_to_wav(input_file):
    try:
        audio = AudioSegment.from_file(input_file)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav_file:
            audio.export(temp_wav_file.name, format="wav")
            return temp_wav_file.name
    except Exception as e:
        print(f"Error converting file: {e}")
        return None

st.title("Music Recommendation using Voice Emotion")
uploaded_file = st.file_uploader("Upload Your Audio File", type=["wav"])
if uploaded_file is not None:
    name = uploaded_file.name
    audio_bytes = uploaded_file.read()
    st.audio(audio_bytes, format='audio/wav')
    s = prediction(name)
    st.write(f"Your mood is {s}")
    dataframe = recommendations(s)
    st.write(dataframe)
