
# EMA helper function for FeatureExtraction.py
def smooth_angle(current, previous, alpha=0.3):
    if previous is None:
        return current
    return alpha * current + (1 - alpha) * previous