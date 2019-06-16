import face_recognition
import os
from PIL import Image


def get_hupu_face():
    if not os.path.exists('./face/hupu/'):
        os.makedirs('./face/hupu/')
    for i in os.listdir('./hupu/images'):
        if not str(i).endswith('jpg'):
            continue
        print("./hupu/images/{}".format(i))
        image = face_recognition.load_image_file("./hupu/images/{}".format(i))
        try:
            face_locations = face_recognition.face_locations(image)
            top, right, bottom, left = face_locations[0]
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.save('./face/hupu/{}'.format(i))
        except:
            continue

def get_SCUT_face():
    if not os.path.exists('./face/SCUT/'):
        os.makedirs('./face/SCUT/')
    for i in os.listdir('./SCUT/images'):
        if not str(i).endswith('jpg'):
            continue
        if str(i)[:2] != 'AF':
            continue

        print("./SCUT/images/{}".format(i))
        image = face_recognition.load_image_file("./SCUT/images/{}".format(i))
        try:
            face_locations = face_recognition.face_locations(image)
            top, right, bottom, left = face_locations[0]
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image.save('./face/SCUT/{}'.format(i))
        except Exception as e:
            continue

get_SCUT_face()
get_hupu_face()