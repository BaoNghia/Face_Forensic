import cv2
import numpy as np
from PIL import Image, ImageDraw
from skimage.transform import SimilarityTransform
from facenet_pytorch import MTCNN, InceptionResnetV1, extract_face
import torch


def read_image(image_path):
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb


def save_image(image_path, image):
    cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))


def plot(image, dets, landms, vis_landms = False, size = 7):
    img_raw = image.copy()
    for b, landm in zip(dets, landms):
        # text = "{:.4f}".format(b[4])
        b = list(map(int, b))
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cx = b[0]
        cy = b[1] + 12
        # cv2.putText(img_raw, text, (cx, cy), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255))
        if vis_landms:
            # landms
            for i, point in enumerate(landm):
                cv2.circle(img_raw, (int(point[0]), int(point[1])), size, (0, 0, 255), -1, cv2.LINE_AA)
    return img_raw

def ShowImages(ImageList, nRows = 1, nCols = 2, WidthSpace = 0.00, HeightSpace = 0.00):
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(nRows, nCols)
    gs.update(wspace=WidthSpace, hspace=HeightSpace) # set the spacing between axes.
    plt.figure(figsize=(20,20))
    for i in range(len(ImageList)):
        ax1 = plt.subplot(gs[i])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')
  
        plt.subplot(nRows, nCols,i+1)
        image = ImageList[i].copy()
        if (len(image.shape) < 3):
            plt.imshow(image, plt.cm.gray)
        else:
            plt.imshow(image)
        plt.title("Image " + str(i+1))
        plt.axis('off')
    plt.show()
    

class FaceDetector:
    def __init__(self, image_size = 320):
        device =  torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(image_size= image_size,
                        thresholds= [0.7, 0.7, 0.8],
                        keep_all= False,
                        device=device)
                        
        self.ARCFACE_SRC = np.array([
            [[122.5, 150.25],
            [197.5, 150.25],
            [160.0, 178.75],
            [137.5, 250.25],
            [182.5, 250.25]]
        ], dtype=np.float32)

    def detect(self, image):
        image = Image.fromarray(image)
        boxes, probs, landmarks = self.mtcnn.detect(image, landmarks=True)
        boxes, probs, landmarks = self.mtcnn.select_boxes(
                boxes, probs, landmarks, image,
        )
        return boxes, probs, landmarks

    def estimate_norm(self, landmarks, image_size=112):
        assert landmarks.shape == (5, 2)
        tform = SimilarityTransform()
        lmk_tran = np.insert(landmarks, 2, values=np.ones(5), axis=1)
        min_M = []
        min_index = []
        min_error = np.inf
        src = self.ARCFACE_SRC
        # if image_size == 112:
        #     src = src
        # else:
        #     src = float(image_size) / 112 * src

        for i in np.arange(src.shape[0]):
            tform.estimate(landmarks, src[i])
            M = tform.params[0:2, :]
            results = np.dot(M, lmk_tran.T)
            results = results.T
            error = np.sum(np.sqrt(np.sum((results - src[i]) ** 2, axis=1)))
            if error < min_error:
                min_error = error
                min_M = M
                min_index = i
        return min_M, min_index

    def norm_crop(self, image, landmark, image_size=112):
        M, pose_index = self.estimate_norm(landmark, image_size)
        warped = cv2.warpAffine(image.copy(), M, (image_size, image_size), borderValue=0.0)
        return warped

def detect(detector, image):
    boxes, probs, landmarks = detector.detect(image)
    if boxes is None:
        return []
    face_list = []
    if len(boxes) > 0:
        for box, landmark in zip(boxes, landmarks):
            landm = landmark.reshape(5, 2).astype('int')
            face_img = detector.norm_crop(image, landm, image_size=320)
            face_list.append(face_img)
    return face_list

def extract(data_path):
    def get_saving_frames_durations(cap, saving_fps, fps):
        """A function that returns the list of durations where to save the frames"""
        s = []
        # get the clip duration by dividing number of frames by the number of frames per second
        clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
        # use np.arange() to make floating-point steps
        for i in np.arange(0, clip_duration, 1 / saving_fps):
            s.append(i)
        return s

    face_detector = FaceDetector(image_size=320)
    reader = cv2.VideoCapture(data_path)
    # get the FPS of the video
    fps = reader.get(cv2.CAP_PROP_FPS)
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(reader, 1, fps)
    frame_num = 0
    while reader.isOpened():
        success, image = reader.read()
        if not success:
            break
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        frame_duration = frame_num / fps
        if frame_duration >= closest_duration:
            face_list = detect(face_detector, image)
            print(len(face_list))
            try:
                # drop the duration spot from the list, 
                # since this duration spot is already saved
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        frame_num += 1
    reader.release()

if __name__ == "__main__":
    path = "data/tmp/frame0.png"
    face_detector = FaceDetector(image_size=320)
    image = read_image(path)
    boxes, probs, landmarks = face_detector.detect(image)

    img_plotted = plot(image, boxes, landmarks, True)
    save_image("abc.png", img_plotted)

    face_list = []
    for box, landmark in zip(boxes, landmarks):
        landm = landmark.reshape(5, 2).astype('int')
        face_img = face_detector.norm_crop(image, landm, image_size=320)
        face_list.append(face_img)
    
    # for i, face in enumerate(face_list):
    #     save_image(f"detected_face_{i}.png", face)
    
        