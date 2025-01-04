

import numpy as np
import cv2


def read_video(videopath):
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        print("Error: Could not open the video file")
        return None, None, None, None

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = cap.get(cv2.CAP_PROP_FPS)

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, frame_width, frame_height, frame_rate

def detectar_semaforo_rojo(frame, en_rojo, historico):
    # Detecta ventana del semáforo y determina si está en rojo
    height, width, _ = frame.shape
    start_x = width // 10
    start_y = height // 2
    light_red = (170, 70, 50) 
    dark_red = (255, 255, 255)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    red_mask = cv2.inRange(hsv_frame, light_red, dark_red)
    red_segmented = cv2.bitwise_and(hsv_frame, hsv_frame, mask=red_mask)
    window_red_seg = red_segmented[start_y:, start_x:]
    semaforo_rojo = cv2.cvtColor(window_red_seg, cv2.COLOR_HSV2BGR)
    suma = semaforo_rojo.sum(axis=(0, 1))

    is_red = suma[2] > suma[0] + suma[1]  # Más rojo que azul y verde combinados

    historico.append(is_red)
    if len(historico) > 4:
        historico.pop(0)  # Mantener solo los últimos 3 estados

    # Actualizar el estado solo si cambia
    if len(historico) == 4 and all(historico) and en_rojo != True:
        print("El semáforo está en rojo")
        en_rojo = True
    elif len(historico) == 4 and not any(historico) and en_rojo != False:
        print("El semáforo está en verde")
        en_rojo = False

    return en_rojo, historico

def frame_binario_eroded(frame):
    # Extrae la región blanca (la carretera) de la imagen
    light_white = (70, 0, 140)
    dark_white = (120, 60, 190)

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white_mask = cv2.inRange(hsv_frame, light_white, dark_white)
    white_segmented = cv2.bitwise_and(hsv_frame, hsv_frame, mask=white_mask)
    gris = cv2.cvtColor(white_segmented, cv2.COLOR_HSV2BGR)
    gris = cv2.cvtColor(gris, cv2.COLOR_BGR2GRAY)
    _, binarizada = cv2.threshold(gris, 30, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(binarizada, np.ones((20, 6), np.uint8), iterations=1)
    dilated = cv2.dilate(dilated, np.ones((5, 2), np.uint8), iterations=1)
    return dilated

def iniciar_kalman_filter(frame_rate):
    kf = cv2.KalmanFilter(6, 2)  # 6 estados (x, y, vx, vy, ax, ay) y 2 medidas (x, y)

    # Matriz de medición (solo medimos posición: x, y)
    kf.measurementMatrix = np.array([
        [1, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0]], dtype=np.float32)

    # Matriz de transición (considera velocidad y aceleración)
    dt = 1 / frame_rate
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0, 0.5 * dt ** 2, 0],
        [0, 1, 0, dt, 0, 0.5 * dt ** 2],
        [0, 0, 1, 0, dt, 0],
        [0, 0, 0, 1, 0, dt],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]], dtype=np.float32)

    # Covarianza del ruido del proceso (estimación del ruido en los estados)
    kf.processNoiseCov = np.eye(6, dtype=np.float32) * 0.03
    return kf

def track_object(frame, kf, crop_hist, initial_window, limites_cruce, en_rojo, trackeando_abajo_ant, direction):

    if not trackeando_abajo_ant: # Instanciar el filtro de Kalman y el histograma de la región de interés
        kf = iniciar_kalman_filter(frame_rate)

        x, y, w, h = initial_window
        track_window = (x, y, w, h)
        cx = x + w//2
        cy = y + h//2

        kf.statePost = np.array([[cx], [cy], [0], [0], [0], [0]], dtype=np.float32)
        kf.errorCovPost = np.eye(6, dtype=np.float32)

        measurement = np.array([[cx], [cy]], np.float32)
        kf.correct(measurement)

        crop = frame[y:(y + h), x:(x + w)].copy()
        hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        crop_hist = cv2.calcHist([hsv_crop], [0], mask=None, histSize=[180], ranges=[0, 180])
        cv2.normalize(crop_hist, crop_hist, 0, 255, cv2.NORM_MINMAX)

    else: # Seguimiento del objeto
        x, y, w, h = initial_window
        term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 1)

        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        back_proj = cv2.calcBackProject([hsv_frame], [0], crop_hist, [0, 180], 1)
        ret, track_window = cv2.meanShift(back_proj, (x, y, w, h), term_crit)

        x, y, w, h = track_window
        cx = x + w//2
        cy = y + h//2

        measurement = np.array([[cx], [cy]], dtype=np.float32)
        kf.correct(measurement)

        if ((direction == "abajo" and cy > limites_cruce[1]) or (direction == "arriba" and cy < limites_cruce[0])):
            mensaje = f"Coche yendo hacia {direction} ha cruzado. {'Multado.' if en_rojo else ''}"
            return None, None, None, False, mensaje
        
    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    return kf, crop_hist, track_window, True, ""

def find_minimized_bounding_box(binary_image, initial_roi):
    x, y, w, h = initial_roi
    initial_region = binary_image[y:y+h, x:x+w]
    total_black_pixels = np.sum(initial_region == 0)

    # Determinar si hay suficientes píxeles negros para buscar el coche
    threshold = 1_000 
    if total_black_pixels < threshold:
        return

    # Encontrar los píxeles negros dentro del rectángulo inicial
    black_pixels = np.argwhere(initial_region == 0)

    # Determinar los límites mínimo y máximo de los píxeles negros
    min_y, min_x = black_pixels.min(axis=0)
    max_y, max_x = black_pixels.max(axis=0)

    # Calcular el bounding box minimizado
    best_x = x + min_x
    best_y = y + min_y
    best_w = max_x - min_x + 1
    best_h = max_y - min_y + 1

    return (best_x, best_y, best_w, best_h)

def draw_status_on_frame(frame, en_rojo, mensaje):
    # Dibujar un rectángulo en la esquina superior izquierda para indicar el estado del semáforo
    color = (0, 0, 255) if en_rojo else (0, 255, 0)  # Rojo si en_rojo, verde si no
    cv2.rectangle(frame, (0, 0), (150, 150), color, -1)

    # Escribir el mensaje en el centro inferior del frame
    text_position = (frame.shape[1] // 2 - 200, frame.shape[0] - 20)  # Centrar texto
    cv2.putText(frame, mensaje, text_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def main(frames, out = False):

    if out:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        frame_size = (frames[0].shape[1], frames[0].shape[0])
        fps = 20
        out = cv2.VideoWriter("video_output.avi", fourcc, fps, frame_size)

    limites_cruce = (415, 555)

    # Rectángulos de interés (ROI) para los carriles de abajo y arriba
    roi_abajo_verde = (375, 10, 220, 600)
    roi_abajo_rojo = (465, 10, 220, 600)
    roi_arriba_verde = (670, 40, 220, 630)
    roi_arriba_rojo = (758, 320, 234, 310)

    trackeando_abajo = False
    trackeando_abajo_ant = None

    trackeando_arriba = False
    trackeando_arriba_ant = None

    historico_en_rojo = [] # Breve historial de estados del semáforo (4)
    en_rojo = None  # Condición inicial 
    mensaje = ""

    for i, frame in enumerate(frames):
        # Detecta si semáforo en rojo
        en_rojo, historico_en_rojo = detectar_semaforo_rojo(frame, en_rojo, historico_en_rojo)

        # Obtiene imagen binaria de la carretera 
        eroded = frame_binario_eroded(frame)

        # Procesar carril abajo
        if trackeando_abajo: # Cuando se está siguiendo un coche
            x, y, w, h = track_window_abajo
            white_pixels = np.sum(eroded[y:y+h, x:x+w] == 255)
            trackeando_abajo_ant = trackeando_abajo
            if white_pixels / eroded[y:y+h, x:x+w].size > 0.6:
                trackeando_abajo = False
            else:
                kf_abajo, crop_hist_abajo, track_window_abajo, trackeando_abajo, mensaje = track_object(
                    frame, kf_abajo, crop_hist_abajo, track_window_abajo, limites_cruce, en_rojo, trackeando_abajo_ant, "abajo")

        if not trackeando_abajo: # Busca primer bounding box para el coche
            minimized_box_abajo = find_minimized_bounding_box(eroded, roi_abajo_verde if not en_rojo else roi_abajo_rojo)
            if minimized_box_abajo and 1.25 < minimized_box_abajo[3] / minimized_box_abajo[2] < 1.65:
                trackeando_abajo_ant = trackeando_abajo
                kf_abajo, crop_hist_abajo, track_window_abajo, trackeando_abajo, mensaje = track_object(
                    frame, None, None, minimized_box_abajo, limites_cruce, en_rojo, trackeando_abajo_ant, "abajo")

        # Procesar carril arriba
        if trackeando_arriba:
            x, y, w, h = track_window_arriba
            white_pixels = np.sum(eroded[y:y+h, x:x+w] == 255)
            trackeando_arriba_ant = trackeando_arriba
            if white_pixels / eroded[y:y+h, x:x+w].size > 0.6:
                trackeando_arriba = False
            else:
                kf_arriba, crop_hist_arriba, track_window_arriba, trackeando_arriba, mensaje = track_object(
                    frame, kf_arriba, crop_hist_arriba, track_window_arriba, limites_cruce, en_rojo, trackeando_arriba_ant, "arriba")

        if not trackeando_arriba:
            minimized_box_arriba = find_minimized_bounding_box(eroded, roi_arriba_verde if not en_rojo else roi_arriba_rojo)
            if minimized_box_arriba and 1.25 < minimized_box_arriba[3] / minimized_box_arriba[2] < 1.65:
                trackeando_arriba_ant = trackeando_arriba
                kf_arriba, crop_hist_arriba, track_window_arriba, trackeando_arriba, mensaje = track_object(
                    frame, None, None, minimized_box_arriba, limites_cruce, en_rojo, trackeando_arriba_ant, "arriba")

        if en_rojo is None:
           draw_status_on_frame(frame, en_rojo, "")
        if mensaje == "":
            mensaje = "Semaforo en ROJO" if en_rojo else "Semaforo en VERDE"
        draw_status_on_frame(frame, en_rojo, mensaje)

        if out:
            out.write(frame)

        cv2.imshow('Frame', frame)
        if cv2.waitKey(10) == ord('q'):
            break

    if out:
        out.release()
        print("Video guardado.")


if __name__ == "__main__":

    # Leer video
    videopath = "video_reducido.avi"
    frames, frame_width, frame_height, frame_rate = read_video(videopath)

    if frames:
        main(frames)

    cv2.destroyAllWindows()