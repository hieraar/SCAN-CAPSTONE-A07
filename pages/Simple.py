import cv2
import streamlit as st
import numpy as np
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import math
import pandas as pd
import networkx as nx
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
import os
from streamlit_webrtc import webrtc_streamer
import av
import threading
import time
import pickle
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from dotenv import load_dotenv

load_dotenv()

user = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
host = os.getenv("DB_HOST")
port = os.getenv("DB_PORT")
db = os.getenv("DB_NAME")

engine = create_engine(f'postgresql+psycopg2://{user}:{password}@{host}:{port}/{db}')
buffer = []
lock = threading.Lock()

def euclidean_distance(point1, point2):
    return math.sqrt((point1[0] - point2[0]) ** 2 +
                     (point1[1] - point2[1]) ** 2 +
                     (point1[2] - point2[2]) ** 2)

def extract_landmarks_and_distances(frame):
    distance_data=pd.DataFrame({})
    reshaped_data = {}
    # Create a Mediapipe FaceMesh object
    mp_face_mesh = mp.solutions.face_mesh

    # Convert frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_h, img_w = frame.shape[:2]

    # Initialize the FaceMesh model
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6) as face_mesh:

        # Process the image and get the face landmarks
        results = face_mesh.process(frame_rgb)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Initialize lists to store distances to index 1
            euclidean_distances = []
            geodesic_distances = []

            # Create a graph for geodesic distances
            G = nx.Graph()

            # Add nodes (landmarks) to the graph
            for idx, landmark in enumerate(face_landmarks.landmark):
                x = landmark.x 
                y = landmark.y 
                z = landmark.z
                G.add_node(idx, pos=(x, y, z))

            # Add edges based on a predefined set of connections (Delaunay triangulation)
            connections = mp.solutions.face_mesh.FACEMESH_TESSELATION
            for edge in connections:
                p1, p2 = edge
                if p1 in G.nodes and p2 in G.nodes:
                    node1 = G.nodes[p1]
                    node2 = G.nodes[p2]
                    dist = euclidean_distance(node1['pos'], node2['pos'])
                    G.add_edge(p1, p2, weight=dist)

            # Calculate Euclidean and geodesic distances from landmark 1
            landmark_indices = [1, 33, 263, 61, 291, 199, 468, 469, 133, 471, 473, 474, 476, 362, 470, 472, 477, 475]
            ref_point = (face_landmarks.landmark[1].x,
                         face_landmarks.landmark[1].y,
                         face_landmarks.landmark[1].z)

            for idx in landmark_indices:
                # Euclidean distance
                landmark = face_landmarks.landmark[idx]
                x = landmark.x 
                y = landmark.y 
                z = landmark.z
                euclidean_distance_value = euclidean_distance(ref_point, (x, y, z))
                euclidean_distances.append(euclidean_distance_value)
                reshaped_data[f'euclidean_{idx}'] = euclidean_distance_value

                # Geodesic distance
                if idx in G.nodes:
                    try:
                        geodesic_distance_value = nx.shortest_path_length(G, source=1, target=idx, weight='weight', method='dijkstra')
                        geodesic_distances.append(geodesic_distance_value)
                        reshaped_data[f'geodesic_{idx}'] = geodesic_distance_value
                    except nx.NetworkXNoPath:
                        geodesic_distances.append(float('inf'))
                        reshaped_data[f'geodesic_{idx}'] = float('inf')
                else:
                    geodesic_distances.append(float('inf'))
                    reshaped_data[f'geodesic_{idx}'] = float('inf')

    # Create a DataFrame with a single row
    distance_data = pd.DataFrame([reshaped_data])
    
    return distance_data 

@st.cache_resource
def load_local_model(model_path):
    # Load the model from a local path
    base_options = python.BaseOptions(model_asset_path=model_path)
    return base_options

def video_frame_callback(frame):
    global img_container
    img = frame.to_ndarray(format="bgr24")
    with lock:
        img_container["img"] = img

    return frame

@st.cache_resource
def load_model(prediction_path):
    with open(prediction_path, 'rb') as file:
        builded_model = pickle.load(file)
    return builded_model

def prepare_features(data):
    features = data[['euclidean_1', 'euclidean_33', 'euclidean_263', 'euclidean_61', 'euclidean_291', 
                     'euclidean_199', 'euclidean_468', 'euclidean_469', 'euclidean_133', 'euclidean_471', 
                     'euclidean_473', 'euclidean_474', 'euclidean_476', 'euclidean_362', 'euclidean_470', 
                     'euclidean_472', 'euclidean_477', 'euclidean_475', 'geodesic_1', 'geodesic_33', 
                     'geodesic_263', 'geodesic_61', 'geodesic_291', 'geodesic_199', 'geodesic_133', 
                     'geodesic_362']]
    return features

def fetch_data(limit):
    Session = sessionmaker(bind=engine)
    session = Session()

    query = f"SELECT * FROM extraction_data LIMIT {limit}"
    data = pd.read_sql(query, con=engine)
    session.close()
    return data

def random_fetch_data(limit):
    Session = sessionmaker(bind=engine)
    session = Session()

    query = f"SELECT * FROM extraction_data ORDER BY RANDOM() LIMIT {limit}"
    data = pd.read_sql(query, con=engine)
    session.close()
    return data

def delete_data():
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        # Using TRUNCATE to delete all data
        session.execute(text("TRUNCATE TABLE extraction_data"))
        session.commit()
        session.close()
        return True  # Return True if deletion is successful
    except Exception as e:
        session.rollback()
        session.close()
        st.error(f"Error deleting data: {e}")
        return False  # Return False if there's an error

def count_rows_in_table():
    Session = sessionmaker(bind=engine)
    session = Session()

    query = "SELECT COUNT(*) FROM extraction_data"
    row_count = pd.read_sql(query, con=engine).iloc[0, 0]  # Retrieve the count from the query result
    session.close()
    return row_count

def show_results():
    Session = sessionmaker(bind=engine)
    session = Session()

    query = "SELECT * FROM predict_result"
    results = pd.read_sql(query, con=engine)  # Retrieve the count from the query result
    session.close()
    return results

def make_predictions(data, model):
    features = prepare_features(data)  # Skip the 'frame_id' column
    predictions = model.predict(features)
    data['Prediction'] = predictions
    return data

def calculate_percentage_of_ones(predictions):
    count_ones = (predictions == 1).sum()  # Count the number of 1s in predictions
    total_predictions = len(predictions)  # Total predictions
    percentage = (count_ones / total_predictions) * 100  # Calculate percentage

    # Determine message based on percentage
    if percentage < 50:
        message = "Tidak diperlukan assesment autis lanjutan"
    elif percentage == 50:
        message = "Kondisi tidak terdefinisi"
    else:
        message = "Terindikasi memiliki spektrum autis sehingga diperlukan assesment lanjutan"
    
    return percentage, message

model_path = os.path.join(r'C:\Model ML', 'model.task')
prediction_path_knn = os.path.join(r'C:\Model ML', 'knn_model_fix.pkl')
prediction_path_dt = os.path.join(r'C:\Model ML', 'dt_model.pkl')
img_container = {"img": None}
extraction_data=None
percentage_of_ones2=None
percentage_of_ones=None
x=None
mdl=None

st.title("Simple Mode")
preview, data = st.columns([0.45,0.55], gap="medium", vertical_alignment="top")

with preview:
    st.subheader("Camera Preview", divider=False)
    with st.container(height=605, border=True):
        ctx = webrtc_streamer(
        key="viewer",
        video_frame_callback=video_frame_callback,
        )
    with st.container(height=60, border=True):
        count_placeholder2 = st.empty()


with data:
    st.subheader("Input patient data", divider="gray")
    with st.form("user_form"):
        name = st.text_input("Name")
        age = st.number_input("Age", min_value=0, max_value=120, step=1)
        condition = st.selectbox("Condition", ["Unknown", "Autis", "Normal"])
    
        # Submit button
        submit_button = st.form_submit_button("Update data", type='primary')
    
        if submit_button:
            st.session_state["user_data"] = {
                "name": name,
                "age": age,
                "condition": condition,
                "model": mdl,
                "frames": x,
                "result_random": percentage_of_ones2,
                "result_normal": percentage_of_ones
            }
            st.toast(f"Patient data '{name}' updated to session!", icon='✅')

    st.subheader("Prediction", divider="gray")
    left_predict, right_predict = st.columns([0.1,0.9], gap='small')
    with left_predict:
        fetch_and_predict_button=st.button('Predict', type='primary')
    with right_predict:
        check_data_button=st.button("Check data", type='secondary')
    
    normal, random = st.columns(2, gap='small')
    with normal:
        with st.container(height=60,border=True):
            result_placeholder_normal=st.empty()
    with random:
        with st.container(height=60,border=True):
            result_placeholder_random=st.empty()

    st.write("Saran tindakan:")
    normal_m, random_m = st.columns(2, gap='small')
    with normal_m:
        with st.container(height=90,border=True):
            message_placeholder2=st.empty()
    with random_m:
        with st.container(height=90,border=True):
            message_placeholder3=st.empty()
    
    if fetch_and_predict_button: 
        x=500
        predict_model = load_model(prediction_path_dt)
        mdl = "DT"  
        data = fetch_data(x)
        data_random = random_fetch_data(x)
        percentage_of_ones=0
        prediction_result=None
        
        if not data.empty:
            prediction_result = make_predictions(data, predict_model)
            percentage_of_ones, message = calculate_percentage_of_ones(prediction_result['Prediction'])
            result_placeholder_normal.write(f"Normal prediction: {percentage_of_ones:.2f}%")
            message_placeholder2.write(f"Normal prediction: {message}")
        
        if not data_random.empty:
            prediction_result = make_predictions(data_random, predict_model)
            percentage_of_ones2, message2 = calculate_percentage_of_ones(prediction_result['Prediction'])
            result_placeholder_random.write(f"Random prediction: {percentage_of_ones2:.2f}%")
            message_placeholder3.write(f"Random prediction: {message}")

        else:
            result_placeholder_normal.write("No data available.")
            result_placeholder_random.write("No data available.")
            message_placeholder2.write("No data available.")
            message_placeholder3.write("No data available.")

        st.session_state["user_data"] = {
                "name": name,
                "age": age,
                "frames": x,
                "condition": condition,
                "model": mdl,
                "result_random": percentage_of_ones2,
                "result_normal": percentage_of_ones
            }
        
        if prediction_result is not None:
            try:
                user_data = pd.DataFrame([st.session_state["user_data"]])
                user_data.to_sql('predict_result', con=engine, if_exists='append', index=False)
                st.toast("Prediction result successfully saved!")
                results_data = show_results()
            
            except Exception as e:
                st.toast(f"Error saving result: {e}")

        else:
            st.toast("No data available, please execute data extraction first")
        
        delete_data()
        count= count_rows_in_table()
        count_placeholder2.write(f"Total Rows: {count}")

    
    if check_data_button:
        count= count_rows_in_table()
        count_placeholder2.write(f"Total Rows: {count}")


st.divider()
left_up, right_up = st.columns(2, gap='medium')

with left_up:
    st.subheader("Results history")
    with st.container(height=200, border=True):
        results_placeholder = st.empty()
    try:
        results_data = show_results()
        results_placeholder.write(results_data)
    except Exception as e:
        st.toast(f"Error retrieving data: {e}")

while ctx.state.playing:
    with lock:
        img = img_container["img"]
    if img is None:
        continue
    
    extracted_distance_data = extract_landmarks_and_distances(img)

    if not extracted_distance_data.empty:
        buffer.append(extracted_distance_data)

    if len(buffer) >= 20:
        try:
            # Concatenate all DataFrames in the buffer
            all_data = pd.concat(buffer)
            all_data = all_data.drop(columns=['geodesic_468','geodesic_469','geodesic_471','geodesic_473','geodesic_474','geodesic_476','geodesic_470','geodesic_472','geodesic_477','geodesic_475'])

            # Insert data into the PostgreSQL table
            all_data.to_sql('extraction_data', con=engine, if_exists='append', index=False, method='multi')
            
            st.toast("Data successfully inserted into the database!")
            count= count_rows_in_table()
            count_placeholder2.write(f"Total Rows: {count}")

            # Clear the buffer after insertion
            buffer = []
            
        except Exception as e:
            # Display an error message if insertion fails
            st.toast(f"Data insertion failed: {e}")