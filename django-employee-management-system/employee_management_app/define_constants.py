# DEFINE PATHS
PEOPLE_DIR = "people"
CSV_FILE_PATH = "results/attendence.csv"

# DEFINE IF YOU WANT TEXT TO SPEECH
text_to_speech = True

# DEFINE CAMERA NUMBER
n_camera = 0

# FACE RECOGNITION CONSTANTS
FACE_RECOGNITION_THRESHOLD = 0.5 # Lesser the value, more accurate the result
n_face_encoding_jitters = 50
DEFAULT_FACE_BOX_COLOR = (0,0,255)
SUCCESS_FACE_BOX_COLOR = (0,255,0)
UNKNOWN_FACE_BOX_COLOR = (0,0,0)
TEXT_IN_FRAME_COLOR = (0,0,255)
TEXT_IN_SUCCESS_COLOR =(0,255,0)

# EYE BLINK DETECTION CONSTANTS
N_MIN_EYE_MIN = 2
N_MAX_EYE_BLINK = 2
EYE_COLOR = (255, 0, 0)
EAR_ratio_threshold = 0.3 # (EAR - Eye Aspect Ratio)
MIN_FRAMES_EYES_CLOSED = 3