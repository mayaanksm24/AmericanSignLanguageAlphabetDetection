from flask import Flask, render_template, Response, request
import logging
import threading
from function import *
from keras.utils import to_categorical
from keras.models import model_from_json
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

app = Flask(__name__)

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

colors = []
for i in range(0,20):
    colors.append((245,117,16))
print(len(colors))
def prob_viz(res, actions, input_frame, colors,threshold):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0,60+num*40), (int(prob*100), 90+num*40), colors[num], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
        
    return output_frame
# 1. New detection variables
sequence = []
sentence = []
accuracy=[]
predictions = []
threshold = 0.8 
camera_on = False
camera_lock = threading.Lock()
def generate_frames():
    cap = cv2.VideoCapture(0)
    mp_hands = mp.solutions.hands
    actions = ['A', 'B', 'C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']  # Update this list to match your model's output classes
    sequence = []
    sentence = []
    accuracy = []
    predictions = []
    threshold = 0.8

    with mp_hands.Hands(
        # Set up the mediapipe hands model with specified complexity and confidence levels
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
        while cap.isOpened(): # Open the camera stream
            ret, frame = cap.read()
            if not ret:
                logging.error("Failed to capture image from camera")
                break

            cropframe = frame[40:400, 0:300]
            frame = cv2.rectangle(frame, (0, 40), (300, 400), 255, 2)
            image, results = mediapipe_detection(cropframe, hands)
           
            keypoints = extract_keypoints(results)
            if keypoints is not None and keypoints.size == 63:  # Ensure keypoints is valid and of the correct shape
                sequence.append(keypoints)
                if len(sequence) > 50:
                    sequence.pop(0)
                
                if len(sequence) == 50:
                    sequence_array = np.array(sequence)
                    res = model.predict(np.expand_dims(sequence_array, axis=0))[0]
                    predictions.append(np.argmax(res))

                    if np.unique(predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > threshold:
                            if len(sentence) > 0:
                                if actions[np.argmax(res)] != sentence[-1]:
                                    sentence.append(actions[np.argmax(res)])
                                    accuracy.append(f"{res[np.argmax(res)] * 100:.2f}%")
                            else:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(f"{res[np.argmax(res)] * 100:.2f}%")
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(f"{res[np.argmax(res)] * 100:.2f}%")

                    if len(sentence) > 1:
                        sentence = sentence[-1:]
                        accuracy = accuracy[-1:]

            output_text = f"Output: {' '.join(sentence)} {' '.join(accuracy)}"
            cv2.rectangle(frame, (0, 0), (300, 40), (245, 117, 16), -1)
            cv2.putText(frame, output_text, (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                logging.error("Failed to encode frame to JPEG")
                break

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
    cv2.destroyAllWindows()

@app.route('/')     # Define the route for the homepage
def index():
    return render_template('index.html') # Render and return the 'index.html' template when the root URL is accessed

@app.route('/video_feed')   # Define the route for the video feed
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_video', methods=['POST'])    # Define the route to start the video feed
def start_video():
    global camera_on
    with camera_lock:
        camera_on = True
    return "Video started", 200

@app.route('/end_video', methods=['POST'])  # Define the route to stop the video feed
def end_video():
    global camera_on
    with camera_lock:
        camera_on = False
    return "Video stopped", 200

if __name__ == '__main__':
    app.run(debug=True)