import numpy as np
import matplotlib.pyplot as plt
import os

def traiettoria_mista_realistica(
    nome_file='./src/data/TP/raw_data/mio/mista.txt',
    segmenti=[(7, 0), (5, 90), (5, 0), (5, -90), (7, 0)],
    passo=1.0,
    rumore_std=0.15,
    partenza=(5.0, 5.0),
    track_id=1,
    frame_iniziale=780,
    frame_passo=10,
    mostra_grafico=True
):
    x_vals, y_vals = [partenza[0]], [partenza[1]]
    angolo_attuale = 0  # in gradi

    for lunghezza, rotazione in segmenti:
        angolo_attuale += rotazione
        rad = np.deg2rad(angolo_attuale)
        dx = passo * np.cos(rad)
        dy = passo * np.sin(rad)

        for _ in range(lunghezza):
            x_next = x_vals[-1] + dx + np.random.normal(0, rumore_std)
            y_next = y_vals[-1] + dy + np.random.normal(0, rumore_std)
            x_vals.append(x_next)
            y_vals.append(y_next)

    punti = len(x_vals)
    frame_ids = [frame_iniziale + i * frame_passo for i in range(punti)]

    os.makedirs(os.path.dirname(nome_file), exist_ok=True)
    with open(nome_file, 'w') as f:
        for i in range(punti):
            #f.write(f"{frame_ids[i]}\t{float(track_id):.1f}\t{round(x_vals[i], 2)}\t{round(y_vals[i], 2)}\n")
            f.write(f"{frame_ids[i]}\t{track_id:.1f}\t{x_vals[i]:.2f}\t{y_vals[i]:.2f}\n")

    print(f"Traiettoria mista realistica salvata in: {nome_file} ({punti} punti)")

    if mostra_grafico:
        plt.figure()
        plt.plot(x_vals, y_vals, 'o-', label='Svolte realistiche')
        plt.gca().set_aspect('equal')

        padding = 2
        plt.xlim(min(x_vals) - padding, max(x_vals) + padding)
        plt.ylim(min(y_vals) - padding, max(y_vals) + padding)

        plt.grid(True)
        plt.title("Traiettoria mista con svolte realistiche")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.tight_layout()
        plt.show()

traiettoria_mista_realistica()
