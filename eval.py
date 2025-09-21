import argparse
from typing import Tuple

from connect4 import Connect4
from agentes import RandomAgent, DefenderAgent
from metodos import TrainedAgent


def eval_pair(agent_a, agent_b, games: int = 100, rows: int = 6, cols: int = 7, render: bool = False) -> Tuple[int, int, int]:
    """Juega 'games' partidas alternando quién empieza y devuelve (wins_a, wins_b, draws)."""
    wins_a = wins_b = draws = 0
    for i in range(games):
        if i % 2 == 0:
            game = Connect4(rows=rows, cols=cols, agent1=agent_a, agent2=agent_b)
            result = game.play(render=render)
            if result == 1:
                wins_a += 1
            elif result == 2:
                wins_b += 1
            else:
                draws += 1
        else:
            game = Connect4(rows=rows, cols=cols, agent1=agent_b, agent2=agent_a)
            result = game.play(render=render)
            if result == 1:
                wins_b += 1
            elif result == 2:
                wins_a += 1
            else:
                draws += 1
    return wins_a, wins_b, draws


def create_opponent(kind: str, rows: int, cols: int, device: str, model_path: str | None):
    kind = (kind or "random").lower()
    if kind == "random":
        return RandomAgent("Random")
    if kind == "defender":
        return DefenderAgent("Defensor")
    if kind == "trained":
        if not model_path:
            raise ValueError("--opponent trained requiere --opponent_model")
        return TrainedAgent(model_path, state_shape=(rows, cols), n_actions=cols, device=device)
    raise ValueError(f"Oponente desconocido: {kind}")


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    try:
        import torch  # noqa: F401
        import torch.cuda as cuda  # noqa: F401
        return "cuda" if cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def main():
    parser = argparse.ArgumentParser(description="Evaluar un agente entrenado en Conecta 4.")
    parser.add_argument("--model", required=True, help="Ruta .pth del agente entrenado (jugador A)")
    parser.add_argument("--opponent", default="random", choices=["random", "defender", "trained"],
                        help="Tipo de oponente (default: random)")
    parser.add_argument("--opponent_model", help="Ruta .pth si el oponente es 'trained'")
    parser.add_argument("--games", type=int, default=100, help="Cantidad de partidas (default: 100)")
    parser.add_argument("--rows", type=int, default=6)
    parser.add_argument("--cols", type=int, default=7)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"],
                        help="Dispositivo para los agentes entrenados (default: auto)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Renderizar tableros (lento)")
    args = parser.parse_args()

    device = resolve_device(args.device)

    agent_a = TrainedAgent(args.model, state_shape=(args.rows, args.cols), n_actions=args.cols, device=device)
    agent_b = create_opponent(args.opponent, args.rows, args.cols, device, args.opponent_model)

    print(f"Evaluando: {agent_a.name} vs. {agent_b.name} | Juegos: {args.games} | Device: {device}")
    wa, wb, dr = eval_pair(agent_a, agent_b, games=args.games, rows=args.rows, cols=args.cols, render=args.verbose)
    rate_a = wa / args.games
    rate_b = wb / args.games
    rate_d = dr / args.games
    print(f"Resultados -> A gana: {wa} ({rate_a:.1%}) | B gana: {wb} ({rate_b:.1%}) | Empates: {dr} ({rate_d:.1%})")


if __name__ == "__main__":
    main()

