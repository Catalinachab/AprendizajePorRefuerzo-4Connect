#import torch.nn as nn
from agentes import Agent
from utils import *


class Connect4State:
    def __init__(self, player1,player2, colums, rows,is_terminal=False, numero_jugada=0): 
        """
        Inicializa el estado del juego Connect4.
        
        Args:
            Definir qué hace a un estado de Connect4.
        """
        self.board = create_board(rows, colums)
        self.current_player = (player1, 1)
        self.other_player = (player2, 2)
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
        copia = Connect4State(self.current_player[0], self.other_player[0], self.board.shape[1], self.board.shape[0], self.is_terminal,self.numero_jugada)
        copia.board=self.board.copy()
        copia.winner=self.winner
        return copia

    def update_state(self, col_elegida):
        """
        Modifica las variables internas del estado luego de una jugada.

        Args:
            ... (_type_): _description_
            ... (_type_): _description_
        """

        insert_token(self.board, col_elegida, self.current_player[1])
        self.current_player, self.other_player = self.other_player, self.current_player

        game_over, winner = check_game_over(self.board)
        
        if game_over:
            self.is_terminal = True
            self.winner = winner
        
        self.numero_jugada+=1
        
        return

    def __eq__(self, other):
        """
        Compara si dos estados son iguales.
        
        Args:
            other: Otro estado para comparar.
            
        Returns:
            True si los estados son iguales, False en caso contrario.
        """
        jugador1 =  self.current_player == other.current_player
        jugador2 =  self.other_player == other.other_player
        tablero = np.array_equal(self.board, other.board)
        termino = self.is_terminal == other.is_terminal
        ganador = self.winner == other.winner
        
        return jugador1 and jugador2 and tablero and termino and ganador 

        

    def __hash__(self): 
        """
        Genera un hash único para el estado.
        
        Returns:
            Hash del estado basado en el tablero y jugador actual.
        """
        return hash((tuple(map(tuple,self.board)), self.current_player, self.other_player, self.is_terminal, self.winner, self.numero_jugada))
        

    def __repr__(self):
        """
        Representación en string del estado.
        
        """
        return (f"Connect4State(\n"
            f"  board={self.board},\n"
            f"  current_player={self.current_player},\n"
            f"  other_player={self.other_player},\n"
            f"  is_terminal={self.is_terminal},\n"
            f"  winner={self.winner},\n"
            f"  numero_jugada={self.numero_jugada}\n"
            f")")

class Connect4Environment:
    def __init__(self, player1, player2, rows, cols):
        """
        Inicializa el ambiente del juego Connect4.
        
        Args:
            Definir las variables de instancia de un ambiente de Connect4

        """
        self.rows = rows
        self.cols = cols
        self.player1 = player1
        self.player2 = player2
        self.current_state = Connect4State(player1, player2, self.cols, self.rows)
        return

    def reset(self):
        """
        Reinicia el ambiente a su estado inicial para volver a realizar un episodio.
        
        """
        self.current_state = Connect4State(self.player1, self.player2, self.cols, self.rows)
        return self.current_state

    def available_actions(self):
        """
        Obtiene las acciones válidas (columnas disponibles) en el estado actual.
        
        Returns:
            Lista de índices de columnas donde se puede colocar una ficha.
        """
        tablero = self.current_state.board
        res=[]
        for i in range(self.cols):
            if tablero[0][i] == 0:
                res.append(i)
        return res

    def step(self, action):
        """
        Ejecuta una acción.
        El estado es modificado acorde a la acción y su interacción con el ambiente.
        Devuelve la tupla: nuevo_estado, reward, terminó_el_juego?, ganador
        Si terminó_el_juego==false, entonces ganador es None.
        
        Args:
            action: Acción elegida por un agente.
            
        """
        if action in self.available_actions():
            moved=self.current_state.current_player[1]
            self.current_state.update_state(action)
            if self.current_state.is_terminal:
                if self.current_state.winner == moved :
                    reward = 1 
                elif self.current_state.winner != None: #gano el otro jugador
                    reward = -1
                else:
                    reward = 0
            else:
                reward = 0
            return (self.current_state, reward, self.current_state.is_terminal, self.current_state.winner)
        else:
            return (f"Acción inválida: {action}. Las acciones válidas son: {self.available_actions()}")

    def render(self):
        """
        Muestra visualmente el estado actual del tablero en la consola.

        """
        print(self.current_state.board)
        print(f"Turno del jugador {self.current_state.current_player[1]}")
        print("------------------------------")
        return
