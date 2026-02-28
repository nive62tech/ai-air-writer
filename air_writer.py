import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0


def fingers_up(hand_landmarks):
    fingers = []

    # Thumb
    if hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index, Middle, Ring, Pinky
    tips = [8, 12, 16, 20]
    for tip in tips:
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)

    return fingers


while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    if canvas is None:
        canvas = np.zeros_like(frame)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            x = int(hand_landmarks.landmark[8].x * frame.shape[1])
            y = int(hand_landmarks.landmark[8].y * frame.shape[0])

            fingers = fingers_up(hand_landmarks)

            cv2.circle(frame, (x, y), 8, (0, 0, 255), -1)

            # Index finger only → DRAW
            if fingers == [0, 1, 0, 0, 0]:
                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(canvas, (prev_x, prev_y), (x, y), (0, 255, 0), 5)
                prev_x, prev_y = x, y

            # Full palm open → MOVE only
            elif fingers == [1, 1, 1, 1, 1]:
                prev_x, prev_y = 0, 0

            # Index + Middle → CLEAR ALL
            elif fingers == [0, 1, 1, 0, 0]:
                canvas = np.zeros_like(frame)
                prev_x, prev_y = 0, 0

            # Closed fist → RUB (erase area)
            elif fingers == [0, 0, 0, 0, 0]:
                cv2.circle(canvas, (x, y), 40, (0, 0, 0), -1)
                prev_x, prev_y = 0, 0

    else:
        prev_x, prev_y = 0, 0

    frame = cv2.add(frame, canvas)

    cv2.imshow("AI Air Writer - Q to Quit", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()