"""
Debug visualization for face tracking
Shows a point following the face position with a trail on a black canvas
"""

import cv2
import numpy as np
import asyncio
import websockets
import json
from collections import deque

class FaceTrackingDebugger:
    def __init__(self, canvas_size=800, trail_length=50):
        self.canvas_size = canvas_size
        self.trail_length = trail_length
        
        # Trail points storage (deque for efficient append/pop)
        self.trail_points = deque(maxlen=trail_length)
        
        # Current position
        self.current_x = canvas_size // 2
        self.current_y = canvas_size // 2
        
        # Face detection status
        self.face_detected = False
        self.confidence = 0.0
        
        # Create window
        cv2.namedWindow('Face Tracking Debug', cv2.WINDOW_NORMAL)
    
    def normalized_to_canvas(self, norm_x, norm_y):
        """Convert normalized coordinates (-1 to 1) to canvas coordinates"""
        canvas_x = int((norm_x + 1) / 2 * self.canvas_size)
        canvas_y = int((1 - norm_y) / 2 * self.canvas_size)  # Flip Y for display
        
        # Clamp to canvas bounds
        canvas_x = max(0, min(self.canvas_size - 1, canvas_x))
        canvas_y = max(0, min(self.canvas_size - 1, canvas_y))
        
        return canvas_x, canvas_y
    
    def draw_grid(self, canvas):
        """Draw reference grid on canvas"""
        grid_color = (30, 30, 30)
        
        # Draw grid lines
        for i in range(0, self.canvas_size, self.canvas_size // 8):
            cv2.line(canvas, (i, 0), (i, self.canvas_size), grid_color, 1)
            cv2.line(canvas, (0, i), (self.canvas_size, i), grid_color, 1)
        
        # Draw center crosshair
        center = self.canvas_size // 2
        cv2.line(canvas, (center - 20, center), (center + 20, center), (50, 50, 50), 2)
        cv2.line(canvas, (center, center - 20), (center, center + 20), (50, 50, 50), 2)
        
        # Draw quadrant labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(canvas, "(-1, 1)", (10, 30), font, 0.5, (100, 100, 100), 1)
        cv2.putText(canvas, "(1, 1)", (self.canvas_size - 80, 30), font, 0.5, (100, 100, 100), 1)
        cv2.putText(canvas, "(-1, -1)", (10, self.canvas_size - 10), font, 0.5, (100, 100, 100), 1)
        cv2.putText(canvas, "(1, -1)", (self.canvas_size - 80, self.canvas_size - 10), font, 0.5, (100, 100, 100), 1)
        cv2.putText(canvas, "(0, 0)", (center + 10, center - 10), font, 0.5, (150, 150, 150), 1)
    
    def draw_frame(self):
        """Draw the debug canvas"""
        # Create black canvas
        canvas = np.zeros((self.canvas_size, self.canvas_size, 3), dtype=np.uint8)
        
        # Draw grid
        self.draw_grid(canvas)
        
        # Draw trail
        if len(self.trail_points) > 1:
            for i in range(1, len(self.trail_points)):
                # Fade trail from old to new
                alpha = i / len(self.trail_points)
                color = (
                    int(0 * (1 - alpha) + 100 * alpha),
                    int(100 * (1 - alpha) + 200 * alpha),
                    int(255 * (1 - alpha) + 100 * alpha)
                )
                thickness = int(1 + 3 * alpha)
                
                cv2.line(
                    canvas,
                    self.trail_points[i - 1],
                    self.trail_points[i],
                    color,
                    thickness
                )
        
        # Draw current position
        if self.face_detected:
            # Draw outer glow
            cv2.circle(canvas, (self.current_x, self.current_y), 20, (0, 100, 255), 2)
            # Draw main point
            cv2.circle(canvas, (self.current_x, self.current_y), 12, (0, 200, 255), -1)
            # Draw inner highlight
            cv2.circle(canvas, (self.current_x, self.current_y), 6, (100, 255, 255), -1)
        else:
            # No face detected - draw gray circle
            cv2.circle(canvas, (self.current_x, self.current_y), 12, (100, 100, 100), 2)
        
        # Draw info text
        font = cv2.FONT_HERSHEY_SIMPLEX
        status_color = (0, 255, 0) if self.face_detected else (100, 100, 100)
        status_text = "Face Detected" if self.face_detected else "No Face"
        
        cv2.putText(canvas, status_text, (10, self.canvas_size - 80), font, 0.7, status_color, 2)
        
        if self.face_detected:
            # Show normalized coordinates
            norm_x = (self.current_x / self.canvas_size) * 2 - 1
            norm_y = 1 - (self.current_y / self.canvas_size) * 2
            coord_text = f"Position: ({norm_x:.2f}, {norm_y:.2f})"
            cv2.putText(canvas, coord_text, (10, self.canvas_size - 50), font, 0.6, (200, 200, 200), 1)
            
            conf_text = f"Confidence: {self.confidence:.2f}"
            cv2.putText(canvas, conf_text, (10, self.canvas_size - 20), font, 0.6, (200, 200, 200), 1)
        
        cv2.imshow('Face Tracking Debug', canvas)
    
    def update_position(self, face_data):
        """Update position from face tracking data"""
        self.face_detected = face_data.get('face_detected', False)
        
        if self.face_detected:
            pos = face_data['position']
            self.confidence = face_data.get('confidence', 0.0)
            
            # Convert to canvas coordinates
            self.current_x, self.current_y = self.normalized_to_canvas(
                pos['x'], pos['y']
            )
            
            # Add to trail
            self.trail_points.append((self.current_x, self.current_y))
    
    async def connect_and_visualize(self, websocket_url="ws://localhost:8765"):
        """Connect to WebSocket and visualize face tracking"""
        print(f"Connecting to {websocket_url}...")
        print("Press 'q' to quit, 'c' to clear trail")
        
        try:
            async with websockets.connect(websocket_url) as websocket:
                print("Connected! Visualizing face tracking...")
                
                while True:
                    try:
                        # Receive face data
                        message = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                        face_data = json.loads(message)
                        self.update_position(face_data)
                    except asyncio.TimeoutError:
                        pass
                    
                    # Draw frame
                    self.draw_frame()
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('c'):
                        self.trail_points.clear()
                        print("Trail cleared")
        
        except websockets.exceptions.WebSocketException as e:
            print(f"WebSocket error: {e}")
            print("Make sure the face tracking server is running!")
        finally:
            cv2.destroyAllWindows()

async def main():
    debugger = FaceTrackingDebugger(canvas_size=800, trail_length=50)
    await debugger.connect_and_visualize()

if __name__ == "__main__":
    asyncio.run(main())