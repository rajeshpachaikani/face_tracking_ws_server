"""
Face tracking script that detects face position and sends data via WebSocket
Outputs normalized coordinates (-1 to 1) for easy animation control
"""

import cv2
import mediapipe as mp
import asyncio
import websockets
import json
import numpy as np
from datetime import datetime

class FaceTracker:
    def __init__(self, camera_id=0):
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0,  # 0 for close-range, 1 for full-range
            min_detection_confidence=0.5
        )
        
        # Initialize camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Smoothing parameters
        self.smooth_factor = 0.3
        self.prev_x = 0.0
        self.prev_y = 0.0
        
        # WebSocket clients
        self.clients = set()
        
    def normalize_position(self, x, y, frame_width, frame_height):
        """
        Convert pixel coordinates to normalized coordinates (-1 to 1)
        Center of screen is (0, 0)
        """
        norm_x = -((x / frame_width) * 2 - 1)  # Flip X axis (mirror camera)
        norm_y = -((y / frame_height) * 2 - 1)  # Flip Y axis
        return norm_x, norm_y
    
    def smooth_position(self, x, y):
        """Apply exponential smoothing to reduce jitter"""
        smooth_x = self.prev_x + self.smooth_factor * (x - self.prev_x)
        smooth_y = self.prev_y + self.smooth_factor * (y - self.prev_y)
        self.prev_x = smooth_x
        self.prev_y = smooth_y
        return smooth_x, smooth_y
    
    def detect_face(self):
        """Detect face and return normalized position"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_frame)
        
        h, w, _ = frame.shape
        
        if results.detections:
            # Get first detected face
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            # Calculate center of face
            face_center_x = (bbox.xmin + bbox.width / 2) * w
            face_center_y = (bbox.ymin + bbox.height / 2) * h
            
            # Normalize coordinates
            norm_x, norm_y = self.normalize_position(face_center_x, face_center_y, w, h)
            
            # Apply smoothing
            smooth_x, smooth_y = self.smooth_position(norm_x, norm_y)
            
            return {
                "face_detected": True,
                "position": {
                    "x": round(smooth_x, 3),
                    "y": round(smooth_y, 3)
                },
                "raw_position": {
                    "x": round(norm_x, 3),
                    "y": round(norm_y, 3)
                },
                "confidence": round(detection.score[0], 3),
                "timestamp": datetime.now().isoformat()
            }
        else:
            return {
                "face_detected": False,
                "position": {"x": 0.0, "y": 0.0},
                "timestamp": datetime.now().isoformat()
            }
    
    async def register_client(self, websocket):
        """Register a new WebSocket client"""
        self.clients.add(websocket)
        print(f"Client connected. Total clients: {len(self.clients)}")
        try:
            await websocket.wait_closed()
        finally:
            self.clients.remove(websocket)
            print(f"Client disconnected. Total clients: {len(self.clients)}")
    
    async def broadcast_position(self):
        """Continuously detect face and broadcast to all clients"""
        while True:
            # Only process if there are connected clients
            if self.clients:
                face_data = self.detect_face()
                
                if face_data:
                    message = json.dumps(face_data)
                    # Broadcast to all connected clients
                    disconnected_clients = set()
                    for client in self.clients:
                        try:
                            await client.send(message)
                        except websockets.exceptions.ConnectionClosed:
                            disconnected_clients.add(client)
                    
                    # Remove disconnected clients
                    self.clients -= disconnected_clients
                
                # Run at ~30 FPS when active
                await asyncio.sleep(1/30)
            else:
                # No clients connected, check less frequently
                await asyncio.sleep(0.5)
    
    async def websocket_handler(self, websocket):
        """Handle WebSocket connections"""
        await self.register_client(websocket)
    
    def cleanup(self):
        """Release resources"""
        self.cap.release()
        self.face_detection.close()

async def main():
    tracker = FaceTracker(camera_id=0)
    
    # Start WebSocket server
    server = await websockets.serve(
        tracker.websocket_handler,
        "localhost",
        8765
    )
    
    print("Face tracking WebSocket server started on ws://localhost:8765")
    print("Waiting for connections...")
    
    try:
        # Run face detection and broadcasting
        await tracker.broadcast_position()
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        tracker.cleanup()
        server.close()
        await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(main())