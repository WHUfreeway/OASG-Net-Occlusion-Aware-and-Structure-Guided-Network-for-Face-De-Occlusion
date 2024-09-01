import face_alignment
from skimage import io

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)

input = io.imread('1.png')
preds = fa.get_landmarks(input)

print(preds)