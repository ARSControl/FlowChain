import numpy as np
import matplotlib.pyplot as plt
import os

def genera_traiettoria_circolare(
    nome_file='./src/data/TP/raw_data/mio/cerchio.txt',
    punti=30,
    raggio=5.0,
    centro_x=10.0,
    centro_y=10.0,
    track_id=1,
    frame_iniziale=780,
    frame_passo=10,
    mostra_grafico=True
):
    # Calcola angoli equidistanti
    angoli = np.linspace(0, 2 * np.pi, punti, endpoint=False)

    # Calcola le coordinate dei punti sul cerchio
    x_vals = centro_x + raggio * np.cos(angoli)
    y_vals = centro_y + raggio * np.sin(angoli)
    frame_ids = [frame_iniziale + i * frame_passo for i in range(punti)]

    # Scrive su file
    os.makedirs(os.path.dirname(nome_file), exist_ok=True)
    with open(nome_file, 'w') as f:
        for i in range(punti):
            f.write(f"{frame_ids[i]}\t{float(track_id):.1f}\t{round(x_vals[i], 2)}\t{round(y_vals[i], 2)}\n")

    print(f"Traiettoria circolare salvata in: {nome_file}")

    # Visualizza grafico
    if mostra_grafico:
        plt.figure(figsize=(6, 6))
        plt.plot(x_vals, y_vals, 'o-', label=f'Traccia {track_id}')
        for i in range(0, punti, punti // 4):  # etichette su 4 punti
            plt.text(x_vals[i], y_vals[i], str(frame_ids[i]), fontsize=8, ha='right')
        plt.gca().set_aspect('equal')
        plt.title("Traiettoria Circolare")
        plt.xlabel("Posizione X")
        plt.ylabel("Posizione Y")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

# Esegui la funzione
genera_traiettoria_circolare()
