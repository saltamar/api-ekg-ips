from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from scipy.signal import find_peaks

app = FastAPI(title="API Edge-EKG IPS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analizar")
async def analizar_electro(
    edad: int = Form(...),
    sexo: str = Form(...),
    sintoma: str = Form(...),
    duracion_trazado: int = Form(...),
    foto: UploadFile = File(...)
):
    try:
        contenido = await foto.read()
        nparr = np.frombuffer(contenido, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        
        if img is None:
            return {"estado": "error", "mensaje": "La imagen no se pudo leer."}

        desenfoque = cv2.GaussianBlur(img, (5, 5), 0)
        mascara = cv2.adaptiveThreshold(desenfoque, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 21, 10)
        kernel = np.ones((3,3), np.uint8)
        senal_limpia = cv2.dilate(mascara, kernel, iterations=1)

        altura = senal_limpia.shape[0]
        ancho = senal_limpia.shape[1]
        pixeles_seg = ancho / duracion_trazado
        
        senal_1d = []
        for columna in senal_limpia.T:
            blancos = np.where(columna > 0)[0]
            if len(blancos) > 0:
                senal_1d.append(altura - np.median(blancos))
            else:
                senal_1d.append(senal_1d[-1] if len(senal_1d) > 0 else 0)
                
        senal_1d = np.array(senal_1d)
        senal_1d_suave = np.convolve(senal_1d, np.ones(5)/5, mode='same')
        
        altura_min = np.mean(senal_1d_suave) + (np.std(senal_1d_suave) * 0.5)
        picos, _ = find_peaks(senal_1d_suave, distance=0.3 * pixeles_seg, height=altura_min, prominence=15)
        
        if len(picos) < 2:
            return {"estado": "error", "mensaje": "Señal insuficiente. Recorta una línea horizontal clara."}

        # ¡AQUÍ ESTÁ LA CORRECCIÓN DE TRADUCCIÓN!
        rr_segs = np.diff(picos) / pixeles_seg
        fc = int(round(60 / np.mean(rr_segs))) # Convertido a entero normal
        cv_porcentaje = float((np.std(rr_segs) / np.mean(rr_segs)) * 100) # Convertido a float normal
        es_regular = bool(cv_porcentaje < 12.0) # Convertido a booleano normal
        cantidad_latidos = int(len(picos)) # Convertido a entero normal

        alerta = "VERDE"
        diagnostico = "Ritmo Sinusal Normal"

        if sintoma == "💔 Dolor de pecho":
            alerta = "AMARILLA"
            diagnostico = "Posible Síndrome Coronario. Requiere evaluación inmediata."
        elif sintoma == "😵 Desmayo" and (fc < 50 or not es_regular):
            alerta = "ROJA"
            diagnostico = "Riesgo de bloqueo o arritmia grave."
        elif not es_regular:
            alerta = "AMARILLA"
            diagnostico = "Ritmo Irregular (Posible FA)."
        elif fc > 100:
            alerta = "AMARILLA"
            diagnostico = "Taquicardia."
        elif fc < 60:
            alerta = "AMARILLA"
            diagnostico = "Bradicardia."

        return {
            "estado": "exito",
            "datos_paciente": {"edad": edad, "sexo": sexo, "sintoma": sintoma},
            "resultados_ia": {
                "frecuencia_cardiaca_lpm": fc,
                "ritmo_regular": es_regular,
                "latidos_detectados": cantidad_latidos
            },
            "triaje": {
                "nivel_alerta": alerta,
                "diagnostico_sugerido": diagnostico
            }
        }

    except Exception as e:
        return {"estado": "error", "mensaje": str(e)}
