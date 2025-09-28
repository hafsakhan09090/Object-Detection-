from flask import Flask, render_template, request, jsonify, Response, send_from_directory
from ultralytics import YOLO
import cv2
import numpy as np
import base64
import os
import tempfile
from datetime import datetime
from collections import defaultdict
import urllib.request

app = Flask(__name__)

camera = None
video_processor = None
detected_objects = defaultdict(int)

def download_weapon_model():
    """Download a pre-trained weapon detection model"""
    model_url = "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt"
    model_path = "yolov8n_weapon.pt"
    
    if not os.path.exists(model_path):
        print("Downloading YOLO model...")
        urllib.request.urlretrieve(model_url, model_path)
        print("Model is downloaded!")
    return model_path

try:
    model_path = download_weapon_model()
    model = YOLO(model_path)
    print("YOLO model loaded successfully!")
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    model = None

class VideoProcessor:
    def __init__(self, video_path):
        """Initialize video processor with video path"""
        self.video_path = video_path
        self.processing = False
        self.cap = None
        
    def start_processing(self):
        """Start video processing"""
        self.processing = True
        self.cap = cv2.VideoCapture(self.video_path)
        return self.cap.isOpened()
    
    def get_next_frame(self):
        """Get next processed frame"""
        if not self.processing or not self.cap or not self.cap.isOpened():
            return None
            
        success, frame = self.cap.read()
        if not success:
            return None
        results = model(frame, conf=0.3, verbose=False)
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                global detected_objects
                detected_objects[class_name] += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if class_name in ['knife', 'scissors', 'gun']:
                    color = (0, 0, 255)  # Red
                elif class_name in ['person']:
                    color = (255, 0, 0)  # Blue
                elif class_name in ['car', 'truck', 'bus']:
                    color = (0, 255, 255)  # Yellow
                else:
                    color = (0, 255, 0)  # Green
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{class_name} {confidence:.2f}', 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return frame
    
    def stop_processing(self):
        """Stop video processing"""
        self.processing = False
        if self.cap:
            self.cap.release()
            self.cap = None

def generate_camera_frames():
    """Generate frames from webcam"""
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            raise Exception("Could not open webcam")
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        results = model(frame, conf=0.3, verbose=False)
        
        if results[0].boxes is not None:
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                global detected_objects
                detected_objects[class_name] += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if class_name in ['knife', 'scissors', 'gun']:
                    color = (0, 0, 255)
                elif class_name in ['person']:
                    color = (255, 0, 0)
                elif class_name in ['car', 'truck', 'bus']:
                    color = (0, 255, 255)
                else:
                    color = (0, 255, 0)
                
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f'{class_name} {confidence:.2f}', 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

def generate_video_frames(video_path):
    """Generate frames from uploaded video"""
    global video_processor
    
    video_processor = VideoProcessor(video_path)
    
    if not video_processor.start_processing():
        yield b"data: Error: Could not open video\n\n"
        return
    
    while video_processor.processing:
        frame = video_processor.get_next_frame()
        if frame is None:
            break
            
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    video_processor.stop_processing()

@app.route('/')
def index():
    return '''
    <html>
    <head>
        <title>Object Detection System</title>
        <style>
            body { font-family: Arial; margin: 20px; background: #0f0f0f; color: white; }
            .container { max-width: 1200px; margin: 0 auto; }
            .tab { overflow: hidden; border: 1px solid #00ff88; background-color: #1a1a1a; }
            .tab button { background-color: inherit; float: left; border: none; outline: none; cursor: pointer; padding: 14px 20px; transition: 0.3s; color: white; }
            .tab button:hover { background-color: #00ff88; color: black; }
            .tab button.active { background-color: #00ff88; color: black; }
            .tabcontent { display: none; padding: 20px; border: 1px solid #00ff88; border-top: none; }
            .upload-area { border: 2px dashed #00ff88; padding: 30px; text-align: center; margin: 20px 0; border-radius: 10px; cursor: pointer; }
            button { background: #00ff88; color: black; padding: 12px 24px; border: none; border-radius: 5px; cursor: pointer; margin: 5px; }
            .stop-btn { background: #ff4444; color: white; }
            .clear-btn { background: #ffaa00; color: black; }
            .result { margin-top: 20px; }
            .video-container { text-align: center; margin: 20px 0; }
            #liveFeed, #videoFeed { max-width: 100%; border: 2px solid #00ff88; }
            .dashboard { margin: 20px 0; padding: 15px; background: #1a1a1a; border: 1px solid #00ff88; border-radius: 10px; }
            .dashboard h3 { margin-top: 0; }
            .dashboard ul { list-style: none; padding: 0; }
            .dashboard li { padding: 5px 0; }
            .loading { text-align: center; padding: 10px; color: #00ff88; }
            .error { color: red; }
            .success { color: #00ff88; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1><b>OBJECT DETECTION BY TEAM FALCONZ</b></h1>
            <div class="tab">
                <button class="tablinks active" onclick="openTab(event, 'Image')">📷 Image</button>
                <button class="tablinks" onclick="openTab(event, 'Video')">🎥 Video</button>
                <button class="tablinks" onclick="openTab(event, 'Live')">🔴 Live Camera</button>
            </div>
            <!-- Image Detection Tab -->
            <div id="Image" class="tabcontent" style="display: block;">
                <h3>Image Detection</h3>
                <div class="upload-area" onclick="document.getElementById('fileInput').click()">
                    <h3>📷 Upload Image</h3>
                    <p>Click to upload an image</p>
                </div>
                <input type="file" id="fileInput" accept="image/*" hidden>
                <button onclick="analyzeImage()">Detect Objects</button>
                <div id="imageResult"></div>
            </div>
            <!-- Video Detection Tab -->
            <div id="Video" class="tabcontent">
                <h3>Video Detection</h3>
                <div class="upload-area" onclick="document.getElementById('videoInput').click()">
                    <h3>🎥 Upload Video</h3>
                    <p>Click to upload a video file (MP4, AVI, MOV)</p>
                </div>
                <input type="file" id="videoInput" accept="video/*" hidden>
                <button onclick="processVideo()">Start Video Processing</button>
                <button class="stop-btn" onclick="stopVideo()" style="display: none;">⏹ Stop Video</button>
                <div class="video-container">
                    <img id="videoFeed" style="display: none; max-width: 100%;">
                </div>
                <div id="videoResult"></div>
            </div>
            <!-- Live Camera Tab -->
            <div id="Live" class="tabcontent">
                <h3>Live Camera</h3>
                <button onclick="startCamera()">Start Camera</button>
                <button class="stop-btn" onclick="stopCamera()" style="display: none;">⏹ Stop Camera</button>
                <div class="video-container">
                    <img id="liveFeed" style="display: none; max-width: 100%;">
                </div>
            </div>
            <div class="dashboard">
                <h3>📊 Detection Dashboard</h3>
                <ul id="objectList">
                    <li>Loading...</li>
                </ul>
                <button class="clear-btn" onclick="clearDashboard()">🗑️ Clear Dashboard</button>
            </div>
        </div>
        <script>
            let currentVideoStream = null;
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
                updateDashboard(); 
            }

            function updateDashboard() {
                fetch('/get_detections')
                    .then(response => response.json())
                    .then(data => {
                        let html = '';
                        for (let obj in data) {
                            html += `<li>${obj}: ${data[obj]} times</li>`;
                        }
                        if (html === '') html = '<li>No detections yet</li>';
                        document.getElementById('objectList').innerHTML = html;
                    })
                    .catch(error => console.error('Dashboard update error:', error));
            }

            function clearDashboard() {
                if (confirm('Are you sure you want to clear all detection counts?')) {
                    fetch('/clear_detections', {
                        method: 'POST'
                    })
                    .then(response => response.json())
                    .then(data => {
                        if (data.success) {
                            updateDashboard();
                            alert('Dashboard cleared successfully!');
                        } else {
                            alert('Error clearing dashboard: ' + data.error);
                        }
                    })
                    .catch(error => {
                        console.error('Clear dashboard error:', error);
                        alert('Error clearing dashboard');
                    });
                }
            }

            setInterval(updateDashboard, 2000); // Update every 2 seconds

            document.getElementById('fileInput').onchange = function() {
                if (this.files[0]) {
                    document.querySelector('#Image .upload-area h3').textContent = '✅ ' + this.files[0].name;
                }
            };

            async function analyzeImage() {
                const fileInput = document.getElementById('fileInput');
                if (!fileInput.files[0]) {
                    alert('Please select an image first!');
                    return;
                }
                
                const formData = new FormData();
                formData.append('image', fileInput.files[0]);
                
                document.getElementById('imageResult').innerHTML = '<div class="loading">Analyzing image...</div>';
                
                try {
                    const response = await fetch('/detect', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    showImageResult(data);
                    updateDashboard(); // Update after image detection
                } catch (error) {
                    document.getElementById('imageResult').innerHTML = '<p class="error">Error: ' + error + '</p>';
                }
            }
            
            function showImageResult(data) {
                let html = '';
                if (data.success) {
                    html += '<h3>Detection Results:</h3>';
                    html += '<p>Found ' + data.detections.length + ' objects</p>';
                    data.detections.forEach(det => {
                        globalThis.detected_objects = globalThis.detected_objects || {};
                        globalThis.detected_objects[det.object] = (globalThis.detected_objects[det.object] || 0) + 1;
                        html += '<div style="background: #2a2a2a; padding: 10px; margin: 5px 0; border-radius: 5px;">';
                        html += '<strong>' + det.object + '</strong> - Confidence: ' + (det.confidence * 100).toFixed(1) + '%';
                        if (det.object === 'knife' || det.object === 'scissors' || det.object === 'gun') {
                            html += ' <span style="color:red">🚨 WEAPON</span>';
                        }
                        html += '</div>';
                    });
                    if (data.image) {
                        html += '<h3>Annotated Image:</h3>';
                        html += '<img src="' + data.image + '" style="max-width: 100%; border: 2px solid #00ff88;">';
                    } else {
                        html += '<p class="error">No annotated image available</p>';
                    }
                } else {
                    html = '<p class="error">Error: ' + data.error + '</p>';
                }
                document.getElementById('imageResult').innerHTML = html;
            }

            document.getElementById('videoInput').onchange = function() {
                if (this.files[0]) {
                    document.querySelector('#Video .upload-area h3').textContent = '✅ ' + this.files[0].name;
                }
            };

            function processVideo() {
                const videoInput = document.getElementById('videoInput');
                if (!videoInput.files[0]) {
                    alert('Please select a video file first!');
                    return;
                }
                
                const formData = new FormData();
                formData.append('video', videoInput.files[0]);
                
                document.getElementById('videoResult').innerHTML = '<div class="loading">Processing video... Please wait</div>';
                const videoFeed = document.getElementById('videoFeed');
                videoFeed.style.display = 'block';
                videoFeed.src = '';
                document.querySelector('#Video .stop-btn').style.display = 'inline-block';
                
                fetch('/upload_video', {
                    method: 'POST',
                    body: formData
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        document.getElementById('videoResult').innerHTML = '<p style="color: #00ff88;">Video processing started!</p>';
                        videoFeed.src = '/video_feed_stream';
                        currentVideoStream = videoFeed.src;
                    } else {
                        document.getElementById('videoResult').innerHTML = '<p class="error">❌ Error: ' + data.error + '</p>';
                        videoFeed.style.display = 'none';
                        document.querySelector('#Video .stop-btn').style.display = 'none';
                    }
                })
                .catch(error => {
                    document.getElementById('videoResult').innerHTML = '<p class="error">❌ Error: ' + error + '</p>';
                    videoFeed.style.display = 'none';
                    document.querySelector('#Video .stop-btn').style.display = 'none';
                });
            }
            
            function stopVideo() {
                const videoFeed = document.getElementById('videoFeed');
                videoFeed.style.display = 'none';
                videoFeed.src = '';
                document.getElementById('videoResult').innerHTML = '<p>Video processing stopped</p>';
                document.querySelector('#Video .stop-btn').style.display = 'none';
                
                fetch('/stop_video')
                    .then(response => response.json())
                    .then(data => console.log(data.status))
                    .catch(error => console.error('Stop video error:', error));
                
                currentVideoStream = null;
            }

            function startCamera() {
                const liveFeed = document.getElementById('liveFeed');
                liveFeed.style.display = 'block';
                liveFeed.src = '/camera_feed';
                document.querySelector('#Live .stop-btn').style.display = 'inline-block';
            }
            
            function stopCamera() {
                const liveFeed = document.getElementById('liveFeed');
                liveFeed.style.display = 'none';
                liveFeed.src = '';
                document.querySelector('#Live .stop-btn').style.display = 'none';
                fetch('/stop_camera')
                    .then(response => response.json())
                    .then(data => console.log(data.status))
                    .catch(error => console.error('Stop camera error:', error));
            }
        </script>
    </body>
    </html>
    '''

@app.route('/detect', methods=['POST'])
def detect_objects():
    if model is None:
        print("Error: Model not loaded")
        return jsonify({'success': False, 'error': 'Model not loaded'})
    
    if 'image' not in request.files:
        print("Error: No image provided")
        return jsonify({'success': False, 'error': 'No image provided'})
    
    file = request.files['image']
    if file.filename == '':
        print("Error: No file selected")
        return jsonify({'success': False, 'error': 'No file selected'})
    
    try:
        image_data = file.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            print("Error: Could not decode image data")
            return jsonify({'success': False, 'error': 'Could not read image'})
        
        print(f"Image shape: {image.shape} - Processing with YOLO...")
        results = model(image, conf=0.3, verbose=False)
        
        detections = []
        annotated_image = image.copy()
        
        if results[0].boxes is not None:
            print(f"Detected {len(results[0].boxes)} objects")
            for box in results[0].boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                global detected_objects
                detected_objects[class_name] += 1
                
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                color = (0, 0, 255) if class_name in ['knife', 'scissors', 'gun'] else (0, 255, 0)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
                cv2.putText(annotated_image, f'{class_name} {confidence:.2f}', 
                           (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                detections.append({
                    'object': class_name,
                    'confidence': confidence,
                    'bbox': [x1, y1, x2, y2]
                })
        else:
            print("No objects detected in image")
        
        _, buffer = cv2.imencode('.jpg', annotated_image)
        if buffer is None:
            print("Error: Failed to encode annotated image")
            return jsonify({'success': False, 'error': 'Failed to encode annotated image'})
        
        image_base64 = base64.b64encode(buffer).decode('utf-8')
        
        print(f"Returning {len(detections)} detections and annotated image")
        return jsonify({
            'success': True,
            'detections': detections,
            'image': f'data:image/jpeg;base64,{image_base64}'
        })
        
    except Exception as e:
        print(f"Exception in detect_objects: {str(e)}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload and initiate processing"""
    if 'video' not in request.files:
        return jsonify({'success': False, 'error': 'No video provided'})
    
    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'error': 'No file selected'})
    
    try:
        temp_dir = tempfile.gettempdir()
        video_path = os.path.join(temp_dir, 'uploaded_video.mp4')
        file.save(video_path)
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            os.remove(video_path)
            return jsonify({'success': False, 'error': 'Could not open video file'})
        cap.release()
        
        return jsonify({'success': True, 'message': 'Video uploaded successfully'})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/video_feed_stream')
def video_feed_stream():
    """Stream processed video frames"""
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, 'uploaded_video.mp4')
    
    if not os.path.exists(video_path):
        return Response("Video not found", status=404)
    
    return Response(generate_video_frames(video_path),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/camera_feed')
def camera_feed():
    """Stream live camera frames"""
    return Response(generate_camera_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_video')
def stop_video():
    """Stop video processing"""
    global video_processor
    if video_processor:
        video_processor.stop_processing()
        video_processor = None
    
    temp_dir = tempfile.gettempdir()
    video_path = os.path.join(temp_dir, 'uploaded_video.mp4')
    if os.path.exists(video_path):
        os.remove(video_path)
    
    return jsonify({'status': 'video stopped'})

@app.route('/stop_camera')
def stop_camera():
    """Stop camera stream"""
    global camera
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({'status': 'camera stopped'})

@app.route('/get_detections')
def get_detections():
    """Return current count of detected objects"""
    global detected_objects
    return jsonify(dict(detected_objects))

@app.route('/clear_detections', methods=['POST'])
def clear_detections():
    """Clear all detection counts"""
    global detected_objects
    try:
        detected_objects.clear()
        return jsonify({'success': True, 'message': 'Dashboard cleared successfully'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("Object Detection System Ready!")
    app.run(debug=True, host='0.0.0.0', port=5000)