import cv2
import numpy as np
from tensorflow.keras.models import load_model

# load the pre-trained model from the final output of  'arm detection.ipynb'
model = load_model('model',compile=False)

# read test video
video_path = 'filepath'
cap = cv2.VideoCapture(video_path)

# get the frame
fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# create output test video filepath
output_path = 'filepath'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

window_size = 10      
stride = 1 
predictions = []


def preprocess_frame(frame):

    # normalization
    normalized_frame = frame / 255.0
    resized_frame = cv2.resize(normalized_frame, (224,224))
    # channel adjustment
    normalized_frame = np.expand_dims(resized_frame, axis=-1)  # 增加通道维度

    # you could add other augmentations if possible

    return normalized_frame
import cv2

def draw_prediction(frame, class_label,entity):
    # draw prediction on frame
    # labels_new = ["left", "right","single",'two']

    text = f"Prediction: {class_label},{entity}"
    org = (5, 50)  # text axis
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    color = (0, 255, 0)  # text color（based on BGR format）
    thickness = 4  # front thickness
    line_type = cv2.LINE_AA  

    
    cv2.putText(frame, text, org, font, font_scale, color, thickness, line_type)

    return frame

# process frame
frame_counter = 0
while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    frame_counter += 1
    if frame_counter % stride != 0:
        continue  

    # preprocess frame
    processed_frame = preprocess_frame(frame)
    labels_new = ["Careful, Turning Left", "No Visible Hand","Careful, Turning Right","Danger! Put both hands on handlebar",'Normal Riding, Safe!']
    # prediction based on model
    prediction = model.predict(np.expand_dims(processed_frame, axis=0))
    predictions.append(prediction)

    # moving average operation to improve smoothness 
    if len(predictions) >= window_size:
        window_predictions = predictions[-window_size:]
        average_prediction = np.mean(window_predictions, axis=0)

        # get classification 
        # class_index = np.argmax(prediction)
        class_index = np.argmax(average_prediction)
        class_text = labels_new[class_index]

    
        frame = draw_prediction(frame, class_index,class_text)

        # write to file
        out.write(frame)

    
    cv2.imshow('Video Classification', frame)

    # press 'q' during the process if want to
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release
cap.release()
out.release()
cv2.destroyAllWindows()
