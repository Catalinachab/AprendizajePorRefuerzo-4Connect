import argparse
import torch
from connect4 import Connect4
from agentes import RandomAgent, DefenderAgent, HumanAgent
from metodos import TrainedAgent


def create_agent(kind: str, name: str, rows: int, cols: int, device: str, model_path: str | None = None):
    kind = kind.lower()
    if kind == "random":
        return RandomAgent(name)
    if kind == "defender":
        return DefenderAgent(name)
    if kind == "human":
        return HumanAgent(name)
    if kind == "trained":
        if not model_path:
            raise ValueError("Debes especificar --model1/--model2 para agentes 'trained'.")
        return TrainedAgent(model_path=model_path, state_shape=(rows, cols), n_actions=cols, device=device)
    raise ValueError(f"Tipo de agente desconocido: {kind}")


def main():
    parser = argparse.ArgumentParser(description="Jugar Conecta 4 entre dos agentes.")
    parser.add_argument("--rows", "-r", type=int, default=6, help="Filas del tablero (default: 6)")
    parser.add_argument("--cols", "-c", type=int, default=7, help="Columnas del tablero (default: 7)")
    parser.add_argument("--agent1", "-a1", type=str, default="random",
                        choices=["random", "defender", "human", "trained"],
                        help="Tipo de agente 1 (default: random)")
    parser.add_argument("--agent2", "-a2", type=str, default="random",
                        choices=["random", "defender", "human", "trained"],
                        help="Tipo de agente 2 (default: random)")
    parser.add_argument("--model1", type=str, help="Ruta .pth para agent1 si es 'trained'")
    parser.add_argument("--model2", type=str, help="Ruta .pth para agent2 si es 'trained'")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"],
                        help="Dispositivo para modelos 'trained' (default: auto)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Mostrar tablero en cada turno")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    a1 = create_agent(args.agent1, "Agente 1", args.rows, args.cols, device, args.model1)
    a2 = create_agent(args.agent2, "Agente 2", args.rows, args.cols, device, args.model2)

    print(f"Juego: {a1.name} (Jugador 1) vs. {a2.name} (Jugador 2) | Device: {device}")
    juego = Connect4(rows=args.rows, cols=args.cols, agent1=a1, agent2=a2)
    ganador = juego.play(render=args.verbose)

    if ganador is None:
        print("Resultado: Empate (0)")
    elif ganador == 1:
        print(f"Resultado: Gana Jugador 1 - {a1.name}")
    elif ganador == 2:
        print(f"Resultado: Gana Jugador 2 - {a2.name}")
    else:
        print(f"Resultado: {ganador}")


if __name__ == "__main__":
    main()
