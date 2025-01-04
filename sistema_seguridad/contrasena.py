import cv2
import numpy as np
import math

# Contraseña esperada
PASSWORD_CORRECT = ["Cuadrado", "Triangulo", "Hexagono","Triangulo"]

def detect_and_store_figures(frame, detected_figures, final_password):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Suavizar la imagen para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (9, 9), 1)
    edges = cv2.Canny(blurred, 50, 50)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    for contorno in contours:
        area = cv2.contourArea(contorno)
        # Filtrar contornos por tamaño
        if area > 1000:
            cv2.drawContours(frame, [contorno], -1, (255, 0, 0), 3)
            
            epsilon = 0.02 * cv2.arcLength(contorno, True)
            approx = cv2.approxPolyDP(contorno, epsilon, True)
            vertices = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            # Determinar el tipo de figura basado en los vértices
            if vertices == 3:
                figura = "Triangulo"
            elif vertices == 4:
                figura = "Cuadrado"
            elif vertices == 5:
                figura = "Pentagono"
            elif vertices == 6:
                figura = "Hexagono"
            elif vertices > 6:
                figura = "Circulo/Poligono"
            else:
                figura = "Desconocido"

            # Calcular el centroide de la figura
            M = cv2.moments(contorno)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])  # Coordenada x del centroide
                cy = int(M["m01"] / M["m00"])  # Coordenada y del centroide
            else:
                cx, cy = 0, 0

            figura_info = {"tipo": figura, "pos": (cx, cy)}

            # Evitar duplicados: Verifica si la figura ya fue almacenada
            figura_existente = False
            for fig in detected_figures:
                if (
                    fig["tipo"] == figura
                    and math.dist(fig["pos"], figura_info["pos"]) < 50  # Tolerancia de 50 px
                ):
                    figura_existente = True
                    break

            if not figura_existente:
                # Añadir figura detectada al registro de detecciones únicas
                detected_figures.append(figura_info)

                # Si se detecta una nueva figura válida, añadirla al intento de contraseña
                final_password.append(figura)

                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, figura, (x+(w//2)-10, y+(h//2)-10),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

    return frame, detected_figures, final_password

def main():
    # Inicializar la cámara
    cap = cv2.VideoCapture(0)
    detected_figures = []  # Lista para registrar figuras únicas detectadas
    final_password = []  # Intento actual de contraseña

    # Bandera para indicar el estado de la contraseña
    password_status = ""

    while True:
        # Capturar frame por frame
        ret, frame = cap.read()
        if not ret:
            print("No se pudo acceder a la cámara.")
            break

        # Detectar figuras y construir intento de contraseña
        processed_frame, detected_figures, final_password = detect_and_store_figures(frame, detected_figures, final_password)

        # Verificar si se ha completado el intento de contraseña
        if len(final_password) == 4:
            if final_password == PASSWORD_CORRECT:
                password_status = "Contrasena correcta!"
                print(password_status)
                # Mostrar mensaje en la ventana
                cv2.putText(processed_frame, password_status, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("Deteccion de Figuras", processed_frame)
                cv2.waitKey(3000)  # Esperar 3 segundos para mostrar el mensaje
                break
            else:
                password_status = "Contrasena incorrecta."
                print(password_status)
                # Reiniciar para nuevo intento
                final_password.clear()

        # Mostrar el progreso actual y estado de la contraseña en el frame
        cv2.putText(processed_frame, f"Progreso: {final_password}", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if password_status:
            cv2.putText(processed_frame, password_status, (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Mostrar el frame con las figuras detectadas y el estado actual
        cv2.imshow("Deteccion de Figuras", processed_frame)

        # Salir si se presiona 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Programa terminado manualmente.")
            break

    # Liberar los recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
