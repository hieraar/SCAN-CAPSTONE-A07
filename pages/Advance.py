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

def draw_landmarks_on_image(rgb_image, detection_result):
  mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(mp_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())

  return annotated_image

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
        min_detection_confidence=min_detection_value,
        min_tracking_confidence=min_tracking_value) as face_mesh:

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

model_path = os.path.join(r'C:\Model ML', 'model.task')
prediction_path_knn = os.path.join(r'C:\Model ML', 'knn_model_fix.pkl')
prediction_path_dt = os.path.join(r'C:\Model ML', 'dt_model.pkl')
img_container = {"img": None}
extraction_data=None

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


#Layouting starts here 

st.title("Advance Mode")
preview, data = st.columns([0.45,0.55], gap="medium", vertical_alignment="top")

with preview:
    st.subheader("Camera Preview", divider=False)
    with st.container(height=627, border=True):
        ctx = webrtc_streamer(
        key="viewer",
        video_frame_callback=video_frame_callback,
        )
    with st.container(height=60, border=True):
        count_placeholder2 = st.empty()

with data:
    
    st.subheader("Extraction Settings", divider="gray")
    left, right = st.columns(2)
    with left:
        min_detection_value= st.slider(
        "Min Detection Confidence", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6, 
        step=0.1
        )

    with right:
        min_tracking_value = st.slider(
        "Min Tracking Confidence", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.6, 
        step=0.1
        )
    st.subheader("Extracted Data", divider="gray")
    with st.container(height=120, border=True):
        table_placeholder=st.empty() 
        message_placeholder= st.empty()  
        
    percentage_of_ones2=None
    percentage_of_ones=None
    x=None
    mdl=None

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
        


st.divider()

df=None

st.title("Data and Prediction")

slider, result = st.columns([0.55,0.45], gap="medium", vertical_alignment="top")



count=0
with slider:
    slider_left, slider_right = st.columns([0.12,0.88], gap="small")
    with slider_left:
        count_button=st.button(f"Count")
        count_placeholder = st.empty()
        if count_button:
            try:
                count= count_rows_in_table()
                count_placeholder.write(f"Rows: {count}")
                count_placeholder2.write(f"Rows: {count}")
            except Exception as e:
                st.write(f"Error :{e}")
        
    with slider_right:
        x = st.slider('Select number of rows to fetch', min_value=100, max_value=1000, value=500, step=100)
    with st.container(height=190, border=True):
        data_fetch=st.empty()

    model_choice = st.selectbox("Choose ML model:", ("K-Nearest Neighbor", "Decision Tree"),index=1, placeholder="Select model...")


with result:
    if model_choice == "K-Nearest Neighbor":
        predict_model = load_model(prediction_path_knn)
        mdl = "KNN"
    if model_choice == "Decision Tree":
        predict_model = load_model(prediction_path_dt)
        mdl = "DT"

    st.write("Make a prediction:")
    left_predict, right_predict = st.columns([0.125,0.875], gap='small')
    with left_predict:
        fetch_and_predict_button=st.button('Predict', type='primary')
    with right_predict:
        save_predict_button=st.button("Save Result", type='secondary')
    
    normal, random = st.columns(2, gap='small')
    with normal:
        with st.container(height=60,border=True):
            result_placeholder_normal=st.empty()
    with random:
        with st.container(height=60,border=True):
            result_placeholder_random=st.empty()

    st.write("Saran tindakan:")
    with st.container(height=60,border=True):
        message_placeholder2=st.empty()
    st.write("Delete data from table:")
    delete_all=st.button("Delete All Data", type='secondary')

    if delete_all:
        delete_data()
        st.toast("All data deleted successfully.")
    

    if fetch_and_predict_button:   

        data = fetch_data(x)
        data_random = random_fetch_data(x)
        percentage_of_ones=0
        

        if not data.empty:
            data_fetch.write(data)
            prediction_result = make_predictions(data, predict_model)
            percentage_of_ones, message = calculate_percentage_of_ones(prediction_result['Prediction'])
            result_placeholder_normal.write(f"Normal predictions: {percentage_of_ones:.2f}%")
            message_placeholder2.write(message)
        
        if not data_random.empty:
            data_fetch.write(data_random)
            prediction_result = make_predictions(data_random, predict_model)
            percentage_of_ones2, message2 = calculate_percentage_of_ones(prediction_result['Prediction'])
            result_placeholder_random.write(f"Random predictions: {percentage_of_ones2:.2f}%")

    
        else:
            result_placeholder_normal.write("No data available.")
            result_placeholder_random.write("No data available.")
            message_placeholder2.write("No data available.")
            data_fetch.write("No data available.")

        st.session_state["user_data"] = {
                "name": name,
                "age": age,
                "frames": x,
                "condition": condition,
                "model": mdl,
                "result_random": percentage_of_ones2,
                "result_normal": percentage_of_ones
            }
        

    
    if save_predict_button:
        try:
            user_data = pd.DataFrame([st.session_state["user_data"]])
            user_data.to_sql('predict_result', con=engine, if_exists='append', index=False)
            st.toast("Prediction result successfully saved!")
            results_data = show_results()
        
        except Exception as e:
            st.toast(f"Error saving result: {e}")


st.divider()
left_up, right_up = st.columns(2, gap='medium')



with left_up:
    st.subheader("File Upload")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df_upload = pd.read_csv(uploaded_file, index_col=0)
        
    insert = st.button("Insert to Database")
    if insert:
        df_upload.to_sql('extraction_data', con=engine, if_exists='replace', index=False, method='multi')
        st.toast("Data successfully inserted into the database!")
        count= count_rows_in_table()
        count_placeholder.write(f"Rows: {count}")
        df_upload= []


with right_up:
    st.subheader("View Results")
    # refresh = st.button("Refresh data")
    with st.container(height=200, border=True):
        results_placeholder = st.empty()

    
    try:
        results_data = show_results()
        results_placeholder.write(results_data)
    except Exception as e:
        st.toast(f"Error retrieving data: {e}")

    # if refresh:
    #     try:
    #         results_data = show_results()
    #         results_placeholder.write(results_data)
        
    #     except Exception as e:
    #         st.toast(f"Error retrieving data: {e}")

    

while ctx.state.playing:
    with lock:
        img = img_container["img"]
    if img is None:
        continue
    
    extracted_distance_data = extract_landmarks_and_distances(img)
    table_placeholder.table(extracted_distance_data)

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
            count_placeholder.write(f"Rows: {count}")
            count_placeholder2.write(f"Total Rows: {count}")

            # Clear the buffer after insertion
            buffer = []
            
        except Exception as e:
            # Display an error message if insertion fails
            message_placeholder.error(f"Data insertion failed: {e}")
