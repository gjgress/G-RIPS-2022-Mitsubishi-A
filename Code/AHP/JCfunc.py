
def MML(delta_d, delta_h, speed):
    d_threshold = 30
    h_threshold = 5
    if speed == 0:
        return True
    elif (delta_d > d_threshold) and (delta_h < h_threshold):
        return True
    else:
        return False