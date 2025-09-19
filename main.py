from connect4 import *
from metodos import *
import torch
from agentes import *
from entrenar import *
import os

juego = Connect4(6, 7)
ganador = juego.play(render=True)
'''
print(f"El ganador es el jugador {ganador}")
'''
# 1. Crear la red con misma arquitectura
input_dim = (6, 7)  # ejemplo: 2 canales (jugador1, jugador2), tablero 6x7
output_dim = 7         # 7 columnas posibles
policy_net = DQN(input_dim, output_dim)
policy_net.load_state_dict(torch.load("trained_model_vs_RandomAgent_1000_0.99_1.0_0.1_0.9950.001_128_1000_100.pth"))
policy_net.eval()

trained_agent = TrainedAgent("trained_model_vs_RandomAgent_1000_0.99_1.0_0.1_0.9950.001_128_1000_100.pth", input_dim, 7, device="cpu")
random_agent = RandomAgent("Random")
game = Connect4(agent1=trained_agent, agent2=random_agent)

# 4. Jugar
winner = game.play(render=True)
print("Ganador:", winner)