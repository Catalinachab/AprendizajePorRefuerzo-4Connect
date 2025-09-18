from connect4 import Connect4

juego = Connect4(6, 7)
ganador = juego.play(render=True)
print(f"El ganador es el jugador {ganador}")


