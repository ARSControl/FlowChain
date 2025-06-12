import random
import matplotlib.pyplot as plt
import numpy as np

def genera_traiettoria_lineare(
    nome_file='./src/data/TP/raw_data/mio/retta.txt', # cartella in cui salvare il file
    punti=50,
    num_tracce=1,  
    frame_iniziale=780,
    frame_passo=10,
    seed=10,
    mostra_grafico=True
):
    random.seed(seed)
    np.random.seed(seed)
    
    # Lista per salvare tutti i dati per il grafico
    tutti_frame = []
    tutti_id = []
    tutti_x = []
    tutti_y = []
    
    with open(nome_file, 'w') as f:
        for track_id in range(1, num_tracce + 1):
            # Punto iniziale e direzione per ogni traccia
            x0 = random.uniform(-4.9, -4)
            y0 = random.uniform(-4.9, -4)
            dx = random.uniform(0.01, 0.35)
            dy = random.uniform(0.01, 0.2)
            
            # Determina quando questa traccia inizia a essere visibile
            frame_apparizione = frame_iniziale
            
            # Genera punti per questa traccia
            for i in range(punti):
                # Calcola il frame corrente
                frame = frame_iniziale + i * frame_passo
                
                # Salta se il frame è prima dell'apparizione della traccia
                if frame < frame_apparizione:
                    continue
                    
                # Aggiungi casualità alla presenza della traccia (meno frequente per rendere più realistico)
                if random.random() < 0.1:  # 10% di probabilità di saltare un punto
                    continue
                
                # Calcola posizione# Calcola posizione
                x = round(x0 + i * dx, 2)
                y = round(y0 + i * dy, 2)
                
                # Limita i punti alla mappa 5x5
                x = np.clip(x, -5, 5)
                y = np.clip(y, -5, 5)
                
                # Scrivi nel file
                f.write(f"{frame}\t{track_id}.0\t{x}\t{y}\n")
                
                # Salva per il grafico
                tutti_frame.append(frame)
                tutti_id.append(track_id)
                tutti_x.append(x)
                tutti_y.append(y)
    
    print(f"Traiettoria scritta in '{nome_file}'")
    
    if mostra_grafico:
        plt.figure(figsize=(8,8))
        plt.xlim(-5, 5)
        plt.ylim(-5, 5)
        # Disegna ogni traccia con un colore diverso
        for track_id in range(1, num_tracce + 1):
            indici = [i for i, id_val in enumerate(tutti_id) if id_val == track_id]
            if indici:
                x_traccia = [tutti_x[i] for i in indici]
                y_traccia = [tutti_y[i] for i in indici]
                frame_traccia = [tutti_frame[i] for i in indici]
                
                plt.plot(x_traccia, y_traccia, marker='o', linestyle='-', 
                         label=f"Traccia {track_id}", alpha=0.7)
                
                # Aggiungi etichette di frame su alcuni punti
                for i in range(0, len(x_traccia), len(x_traccia)//3 + 1):
                    plt.text(x_traccia[i], y_traccia[i], str(frame_traccia[i]), 
                             fontsize=8, ha='right')
        
        plt.title("Traiettoria Lineare")
        plt.xlabel("Posizione X")
        plt.ylabel("Posizione Y")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

# Esegui la funzione con una sola traiettoria
genera_traiettoria_lineare(frame_iniziale=780, num_tracce=1)