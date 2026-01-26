import numpy as np

def calculate_angle(point1, point2, point3):
    """Calculate angle between three points"""
    a = np.array(point1)
    b = np.array(point2)  # vertex point
    c = np.array(point3)
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle > 180.0:
        angle = 360-angle
    
    return angle