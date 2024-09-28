import pygame
import sys
import numpy as np
from tensorflow.keras.models import load_model
import cv2

# Constants
WINDOWSIZEX = 640
WINDOWSIZEY = 480
BOUNDARYINC = 5
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
IMAGESAVE = False  # Set this to True if you want to save images
MODEL = load_model("C:/Users/sasir/OneDrive/Desktop/digit_recog/best_model.keras")
LABELS = {
    0: "Zero", 1: "One", 2: "Two", 3: "Three",
    4: "Four", 5: "Five", 6: "Six", 7: "Seven",
    8: "Eight", 9: "Nine"
}

# Initialize pygame
pygame.init()
FONT = pygame.font.Font("FreeSansBold.ttf", 18)
DISPLAYSURF = pygame.display.set_mode((WINDOWSIZEX, WINDOWSIZEY))
pygame.display.set_caption("Digit Board")

iswriting = False
number_xcord = []
number_ycord = []

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        elif event.type == pygame.MOUSEMOTION and iswriting:
            xcord, ycord = event.pos
            pygame.draw.circle(DISPLAYSURF, WHITE, (xcord, ycord), 4, 0)
            number_xcord.append(xcord)
            number_ycord.append(ycord)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            iswriting = True
        elif event.type == pygame.MOUSEBUTTONUP:
            iswriting = False
            number_xcord = sorted(number_xcord)
            number_ycord = sorted(number_ycord)

            rect_min_x, rect_max_x = max(number_xcord[0] - BOUNDARYINC, 0), min(WINDOWSIZEX, number_xcord[-1] + BOUNDARYINC)
            rect_min_y, rect_max_y = max(number_ycord[0] - BOUNDARYINC, 0), min(WINDOWSIZEY, number_ycord[-1] + BOUNDARYINC)

            number_xcord = []
            number_ycord = []

            # Capture and process the image for prediction
            img_arr = np.array(pygame.PixelArray(DISPLAYSURF))[rect_min_x:rect_max_x, rect_min_y:rect_max_y].T.astype(np.float32)
            image = cv2.resize(img_arr, (28, 28))
            image = np.pad(image, (10, 10), 'constant', constant_values=0)
            image = cv2.resize(image, (28, 28)) / 255

            if IMAGESAVE:
                cv2.imwrite('image.png', img_arr)  # Optionally save the captured image

            # Perform prediction
            prediction = MODEL.predict(image.reshape(1, 28, 28, 1))
            label = str(LABELS[np.argmax(prediction)])

            # Draw bounding box around detected area
            pygame.draw.rect(DISPLAYSURF, RED, (rect_min_x, rect_min_y, rect_max_x - rect_min_x, rect_max_y - rect_min_y), 2)

            # Render predicted label on the screen
            textSurface = FONT.render(label, True, RED, WHITE)
            textRecObj = textSurface.get_rect()
            textRecObj.center = (rect_min_x + (rect_max_x - rect_min_x) // 2, rect_max_y + 20)  # Adjust text position
            DISPLAYSURF.blit(textSurface, textRecObj)

        elif event.type == pygame.KEYDOWN:
            if event.unicode == "n":
                DISPLAYSURF.fill(BLACK)  # Clear screen on 'n' key press

    pygame.display.update()
