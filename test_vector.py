import serial
import time
import json

PUERTO = '/dev/ttyUSB0'
BAUDIOS = 115200

def enviar_vector_estado():
    ser = None
    try:
        print(f"🔌 Conectando al Centinela en {PUERTO}...")
        ser = serial.Serial(PUERTO, BAUDIOS, timeout=1)
        
        print("⏳ Esperando reinicio del chip (3 segundos)...")
        time.sleep(3) 
        
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        # El Vector de Estado (Simulando el proyecto SAGUT)
        vector_estado = {
            "type": "state_update",
            "physics": {
                "saturation": 0.85,         
                "dissipated_power": 65.0,   
                "gyroscopic_stability": 0.4 
            },
            "topology": {
                "beta_1": 442,              
                "pyramid_stability": 0.69   
            },
            "wisdom": {
                "verdict_code": 2,          
                "narrative": "FIEBRE ESTRUCTURAL"
            }
        }
        
        # Convertir a string JSON y agregar el Enter (\n)
        payload = json.dumps(vector_estado) + "\n"
        
        print(f"📨 Enviando Vector MIC:\n{json.dumps(vector_estado, indent=2)}")
        ser.write(payload.encode('utf-8'))
        
        print("\n👂 Escuchando reacción del hardware...")
        start_time = time.time()
        
        while time.time() - start_time < 3:
            if ser.in_waiting > 0:
                linea = ser.readline().decode('utf-8', errors='replace').strip()
                if linea:
                    print(f"   🤖 Hardware: {linea}")

    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        if ser and ser.is_open:
            ser.close()
            print("🔌 Conexión cerrada.")

if __name__ == "__main__":
    enviar_vector_estado()