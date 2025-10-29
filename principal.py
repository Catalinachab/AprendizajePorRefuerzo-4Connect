#import torch.nn as nn
from agentes import Agent
from utils import *


class Connect4State:
    def __init__(self, colums, rows, is_terminal=False, numero_jugada=0): 
        """
        Inicializa el estado del juego Connect4.
        
        Args:
            Definir qué hace a un estado de Connect4.
        """
        self.board = create_board(rows, colums)
        self.current_player = 1  # Jugador actual: 1 o 2
        self.is_terminal = is_terminal
        self.winner = None
        self.numero_jugada = numero_jugada
        return

    def copy(self):  
        """
        Crea una copia profunda del estado actual.
        
        Returns:
            Una nueva instancia de Connect4State con los mismos valores.
        """
        copia = Connect4State(self.board.shape[1], self.board.shape[0], self.is_terminal, self.numero_jugada)
        copia.board = self.board.copy()
        copia.current_player = self.current_player
        copia.winner = self.winner
        return copia

    def update_state(self, col_elegida):
        """
        Modifica las variables internas del estado luego de una jugada.

        Args:
            ... (_type_): _description_
            ... (_type_): _description_
        """

        insert_token(self.board, col_elegida, self.current_player)
        # Alternar jugador: 1 -> 2, 2 -> 1
        self.current_player = 3 - self.current_player

        game_over, winner = check_game_over(self.board)
        
        if game_over:
            self.is_terminal = True
            self.winner = winner
        
        self.numero_jugada += 1
        
        return

    def __eq__(self, other):
        """
        Compara si dos estados son iguales.
        
        Args:
            other: Otro estado para comparar.
            
        Returns:
            True si los estados son iguales, False en caso contrario.
        """
        jugador = self.current_player == other.current_player
        tablero = np.array_equal(self.board, other.board)
        termino = self.is_terminal == other.is_terminal
        ganador = self.winner == other.winner
        
        return jugador and tablero and termino and ganador 

        

    def __hash__(self): 
        """
        Genera un hash único para el estado.
        
        Returns:
            Hash del estado basado en el tablero y jugador actual.
        """
        return hash((tuple(map(tuple, self.board)), self.current_player, self.is_terminal, self.winner, self.numero_jugada))
        

    def __repr__(self):
        """
        Representación en string del estado.
        
        """
        return (f"Connect4State(\n"
            f"  board={self.board},\n"
            f"  current_player={self.current_player},\n"
            f"  is_terminal={self.is_terminal},\n"
            f"  winner={self.winner},\n"
            f"  numero_jugada={self.numero_jugada}\n"
            f")")

class Connect4Environment:
    def __init__(self, rows, cols):
        """
        Inicializa el ambiente del juego Connect4.
        
        Args:
            rows: Número de filas del tablero
            cols: Número de columnas del tablero
        """
        self.rows = rows
        self.cols = cols
        self.state = Connect4State(self.cols, self.rows)
        return

    def reset(self):
        """
        Reinicia el ambiente a su estado inicial para volver a realizar un episodio.
        
        """
        self.state = Connect4State(self.cols, self.rows)
        return self.state

    def available_actions(self):
        """
        Obtiene las acciones válidas (columnas disponibles) en el estado actual.
        
        Returns:
            Lista de índices de columnas donde se puede colocar una ficha.
        """
        tablero = self.state.board
        res = []
        for i in range(self.cols):
            if tablero[0][i] == 0:
                res.append(i)
        return res

    def step(self, action):
        """
        Ejecuta una acción.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Devuelve la tupla: nuevo_estado, reward, terminó_el_juego?, info_dict
        info_dict contiene {"winner": winner} donde winner es None, 1, o 2
        
        Args:
            action: Acción elegida por un agente.
            
        """
        if action in self.available_actions():
            moved = self.state.current_player
            self.state.update_state(action)
            if self.state.is_terminal:
                if self.state.winner == moved:
                    reward = 1 
                elif self.state.winner != None:  # gano el otro jugador
                    reward = -1
                else:
                    reward = 0
            else:
                reward = 0
            # Retornar info como diccionario para compatibilidad con template
            info = {"winner": self.state.winner}
            return (self.state, reward, self.state.is_terminal, info)
        else:
            return (f"Acción inválida: {action}. Las acciones válidas son: {self.available_actions()}")

    def render(self):
        """
        Muestra visualmente el estado actual del tablero en la consola.

        """
        print(self.state.board)
        print(f"Turno del jugador {self.state.current_player}")
        print("------------------------------")
        return
