import cv2 as cv
import mediapipe as mp

# sets up webcam
cam = cv.VideoCapture(0)

# initializes hand object
init_hands = mp.solutions.hands
hand = init_hands.Hands()

# initializes drawing object
draw_utils = mp.solutions.drawing_utils

# loop to capture video from the screen and display it on the screen
while True:
    success, img = cam.read()

    # flip image and convert from bgr to rgb
    img = cv.flip(img, 1)
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # stores final image with hand tracking in final_img
    final_img = hand.process(rgb_img)

    # initialize list to store id, x, and y values
    landmarks = []

    # draws landmarks on screen, calculates x and y values, then adds them to the landmarks list
    if final_img.multi_hand_landmarks:
        for detected_hand in final_img.multi_hand_landmarks:
            for id, lm in enumerate(detected_hand.landmark):
                height, width, channels = img.shape
                center_x = int(lm.x * width)
                center_y = int(lm.y * height)
                landmarks.append([id, center_x, center_y])
                print([id, center_x, center_y])
        draw_utils.draw_landmarks(img, detected_hand, init_hands.HAND_CONNECTIONS)

    cv.imshow("Video", img)
    cv.waitKey(15)
