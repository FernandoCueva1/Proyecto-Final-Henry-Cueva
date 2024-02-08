import cv2
import torch
import numpy as np
from sort import Sort
import matplotlib.path as mplPath
import tkinter as tk
from PIL import ImageTk, Image
import threading

def generate_zones(person_coordinates, screen_width, screen_height, num_zones):
    zones = []

    # Calcular el tamaño de las zonas
    zone_width = screen_width // num_zones
    zone_height = screen_height // num_zones

    # Iterar sobre las coordenadas de las personas detectadas
    for person_coord in person_coordinates:
        # Obtener las coordenadas de la persona
        x, y = person_coord

        # Calcular el índice de la zona basado en las coordenadas de la persona
        zone_index_x = x // zone_width
        zone_index_y = y // zone_height

        # Calcular los límites de la zona
        left = max(0, zone_index_x * zone_width)
        right = min(screen_width, (zone_index_x + 1) * zone_width)
        top = max(0, zone_index_y * zone_height)
        bottom = min(screen_height, (zone_index_y + 1) * zone_height)

        # Crear la zona alrededor de la persona
        zone = np.array([
            [left, top],
            [right, top],
            [right, bottom],
            [left, bottom],
        ])
        center = (left + (right - left) // 2, top + (bottom - top) // 2)
        zones.append((zone, center))

    return zones

def get_center(bbox):
    center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
    return center

def load_model():
    model = torch.hub.load("ultralytics/yolov5", model="yolov5n", pretrained=True)
    return model

def get_bboxes(preds):
    df = preds.pandas().xyxy[0]
    df = df[df["confidence"] >= 0.50]
    df = df[df["name"] == "person"]
    return df[["xmin", "ymin", "xmax", "ymax"]].values.astype(int)

def is_valid_detection(xc, yc, zone):
    return mplPath.Path(zone).contains_point((xc, yc))

def detector(cap, output_label):
    paused = threading.Event()  # Mover la inicialización de 'paused' aquí
    model = load_model()
    tracker = Sort()

    while cap.isOpened():
        if not paused.is_set():
            status, frame = cap.read()
            if not status:
                break

            preds = model(frame)
            bboxes = get_bboxes(preds)

            pred_confidences = preds.xyxy[0][:, 4].cpu().numpy()

            trackers = tracker.update(bboxes)

            # Obtener las coordenadas de las personas detectadas
            person_coordinates = [(int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)) for box in trackers]

            # Generar las zonas alrededor de las personas detectadas
            zones = generate_zones(person_coordinates, frame.shape[1], frame.shape[0], 3)

            detections_zones = [0] * len(zones)

            for i, box in enumerate(trackers):
                xc, yc = get_center(box)
                xc, yc = int(xc), int(yc)

                cv2.rectangle(img=frame, pt1=(int(box[0]), int(box[1])), pt2=(int(box[2]), int(box[3])), color=(0, 255, 0), thickness=2)  # Cambiar el color aquí
                cv2.circle(img=frame, center=(xc, yc), radius=5, color=(255, 0, 0), thickness=-1)  # Cambiar el color aquí
                cv2.putText(img=frame, text=f"id: {int(box[4])}, conf: {pred_confidences[i]:.2f}", org=(int(box[0]), int(box[1])), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 255), thickness=2)

                for j, (zone, _) in enumerate(zones):
                    if is_valid_detection(xc, yc, zone):
                        detections_zones[j] += 1

            for j, (zone, center) in enumerate(zones):
                cv2.putText(img=frame, text=f"Area {j+1}: {detections_zones[j]}", org=(center[0] - 30, center[1]), fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(0, 0, 0), thickness=2)
                cv2.polylines(img=frame, pts=[zone], isClosed=True, color=(255, 255, 0), thickness=3)  # Cambiar el color aquí

            # Convertir el frame a formato adecuado para mostrar en un widget Label de tkinter
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)

            # Actualizar la imagen en el widget Label
            output_label.config(image=frame)
            output_label.image = frame

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()

def iniciar_programa(cap, output_label):
    global programa_thread
    programa_thread = threading.Thread(target=detector, args=(cap, output_label))
    programa_thread.start()
    btn_iniciar.config(state=tk.DISABLED, bg="#003366", fg="#FFFFFF", activebackground="#001a33", activeforeground="#FFFFFF")
    btn_detener.config(state=tk.NORMAL, bg="#990000", fg="#FFFFFF", activebackground="#660000", activeforeground="#FFFFFF")
    root.configure(bg="#FFFFFF")  # Cambia el color de fondo de la ventana

def detener_programa():
    cap.release()
    root.quit()

# Crear la interfaz gráfica
root = tk.Tk()
root.title("DETECTOR DE PERSONAS")
root.geometry("800x600")
root.configure(bg="#CCCCCC")  # Cambia el color de fondo de la ventana

# Estilos
font_style = ("Helvetica", 12)

# Marco para los botones
button_frame = tk.Frame(root, bg="#CCCCCC")  # Cambia el color de fondo del marco de botones
button_frame.pack(side=tk.TOP, fill=tk.X)

# Botones
cap = cv2.VideoCapture("D:\Fer\Proyecto Henry Cueva/video.webm")
btn_iniciar = tk.Button(button_frame, text="Iniciar", command=lambda: iniciar_programa(cap, output_label), font=font_style, bg="#003366", fg="#FFFFFF", activebackground="#001a33", activeforeground="#FFFFFF")
btn_iniciar.pack(side=tk.LEFT, padx=5)
btn_detener = tk.Button(button_frame, text="Salir", command=detener_programa, font=font_style, bg="#990000", fg="#FFFFFF", activebackground="#660000", activeforeground="#FFFFFF")
btn_detener.pack(side=tk.LEFT, padx=5)

# Etiqueta para mostrar la salida de la cámara
output_label = tk.Label(root, bg="#F0F0F0")  # Cambia el color de fondo de la etiqueta de salida
output_label.pack(pady=10)

root.mainloop()


