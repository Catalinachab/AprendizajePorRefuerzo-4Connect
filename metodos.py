
from principal import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import random
from agentes import Agent

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim): 
        """
        Inicializa la red neuronal DQN para el aprendizaje por refuerzo.
        
        Args:
            input_dim: Dimensión de entrada (número de features del estado).
            output_dim: Dimensión de salida (número de acciones posibles).
        """
        super().__init__()
        c=2 #jugadores
        h=input_dim[0]
        w = input_dim[1]
        self.features = nn.Sequential(
        nn.Conv2d(h, w, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
        nn.Flatten(),
        nn.Linear(128 * h * w, 256),
        nn.ReLU(inplace=True),
        nn.Linear(256, output_dim),
        nn.Linear(256, output_dim)  #  Q-values crudos
        )
        
        

    def forward(self, x):
        """
        Pasa la entrada a través de la red neuronal.
        
        Args:
            x: Tensor de entrada.
            
        Returns:
            Tensor de salida con los valores Q para cada acción.
        """
        if x.dim() == 3:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.head(x)
        return x

class DeepQLearningAgent:
    def __init__(self, state_shape: Tuple[int, int], n_actions: int, device: torch.device,
                 gamma: float = 0.99, epsilon: float = 1, epsilon_min: float = 0.1, epsilon_decay: float = 0.995,
                 lr = 1e-3, batch_size: int = 64, memory_size: int = 10_000, target_update_every: int = 1000): 
        """
        Inicializa el agente de aprendizaje por refuerzo DQN.
        
        Args:
            state_shape: Forma del estado (filas, columnas).
            n_actions: Número de acciones posibles.
            device: Dispositivo para computación ('cpu' o 'cuda').
            gamma: Factor de descuento para recompensas futuras.
            epsilon: Probabilidad inicial de exploración.
            epsilon_min: Valor mínimo de epsilon.
            epsilon_decay: Factor de decaimiento de epsilon.
            lr: Tasa de aprendizaje.
            batch_size: Tamaño del batch para entrenamiento.
            memory_size: Tamaño máximo de la memoria de experiencias.
            target_update_every: Frecuencia de actualización de la red objetivo.
        """
        self.device = device
        self.gamma = gamma  
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.learningRate = lr
        self.target_update_every = target_update_every
        self.memory = []  # Memoria de experiencias
        self.memory_size = memory_size
        self.step_count = 0  # Contador de pasos para actualizar la red objetivo
        self.policy_net = DQN(state_shape, n_actions).to(device)
        self.target_net = DQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # La red objetivo no se entrena
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.train_steps = 0

    def preprocess(self, state):
        """
        Convierte el estado del juego a un tensor de PyTorch.
        
        Args:
            state: Estado del juego.
            
        Returns:
            Tensor de PyTorch con el estado aplanado.
        """
        print(state.board + "este es el estado")
        board = state.board
        arr = np.array(board, dtype=np.float32)        # (rows, cols)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, rows, cols]
        return tensor.to(self.device)

    def select_action(self, state, valid_actions): 
        """
        Selecciona una acción usando la política epsilon-greedy.
        
        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.
            
        Returns:
            Índice de la acción seleccionada.
        """
    
        # 1. Exploración
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)

        # 2. Explotación
        with torch.no_grad():
            s_t = self.preprocess(state)        # [rows, cols]
            q_values = self.policy_net(s_t)     # [1, n_actions]: asigna proba a cada posibilidad

            # Enmascarar acciones inválidas
            mask = torch.full_like(q_values, -1e9)  # todas muy bajas
            mask[:, valid_actions] = q_values[:, valid_actions]

            # Seleccionamos el índice de la mejor acción válida
            action = int(torch.argmax(mask, dim=1).item())
            
            return action

    def store_transition(self, s, a, r, s_next, done):
        """
        Almacena una transición (estado, acción, recompensa, siguiente estado, terminado) en la memoria.
        
        Args:
            s: Estado actual.
            a: Acción tomada.
            r: Recompensa obtenida.
            s_next: Siguiente estado.
            done: Si el episodio terminó.
        """
         # Quitamos la dimensión de batch -> [rows, cols]
        s_t = self.preprocess(s).squeeze(0).to(self.device)
        s_next_t = self.preprocess(s_next).squeeze(0).to(self.device)
        self.memory.append(self.Transition(s_t, a, r, s_next_t, done))

    def train_step(self): 
        """
        Ejecuta un paso de entrenamiento usando experiencias de la memoria.
        
        Returns:
            Valor de la función de pérdida si se pudo entrenar, None en caso contrario.
        """
        if len(self.memory) < self.batch_size:
            return None

         # -------- Sample batch --------
        batch = random.sample(self.memory, self.batch_size)
        # Cada s está guardado como [rows, cols]; lo pasamos a [B, 1, rows, cols]
        s_batch      = torch.stack([b.s for b in batch], dim=0).unsqueeze(1).to(self.device)
        s_next_batch = torch.stack([b.s_next for b in batch], dim=0).unsqueeze(1).to(self.device)
        a_batch      = torch.tensor([b.a for b in batch], dtype=torch.long, device=self.device)      # [B]
        r_batch      = torch.tensor([b.r for b in batch], dtype=torch.float32, device=self.device)   # [B]
        done_batch   = torch.tensor([b.done for b in batch], dtype=torch.bool, device=self.device)   # [B]

        # -------- Q(s,a) predicho --------
        q_all = self.policy_net(s_batch)                      # [B, n_actions]
        q_sa  = q_all.gather(1, a_batch.unsqueeze(1)).squeeze(1)  # [B]

        # -------- Target con Double DQN --------
        with torch.no_grad():
            # Online net elige la mejor acción válida en s'
            q_next_online = self.policy_net(s_next_batch)     # [B, n_actions]
            q_next_online_masked = self._mask_q_next(q_next_online, s_next_batch)
            next_acts = q_next_online_masked.argmax(dim=1)    # [B]

            # Target net evalúa esa acción
            q_next_target_all = self.target_net(s_next_batch) # [B, n_actions]
            q_next_target = q_next_target_all.gather(1, next_acts.unsqueeze(1)).squeeze(1)  # [B]

            # y(s) = r + gamma * max_a' Q_target(s', a')  (si no es terminal)
            target = r_batch + (~done_batch).float() * self.gamma * q_next_target  # [B]

            # -------- Loss & step --------
            loss = self.loss_fn(q_sa, target)
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
            self.optimizer.step()

            # -------- Target update --------
            self.grad_steps += 1
            if self.grad_steps % self.target_update_every == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            return float(loss.item())
            

    def update_epsilon(self):
        """
        Actualiza el valor de epsilon para reducir la exploración gradualmente.
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return

class TrainedAgent(Agent):
    def __init__(self, model_path: str, state_shape: tuple, n_actions: int, device='cpu'):
        """
        Inicializa un agente DQN pre-entrenado.
        
        Args:
            model_path: Ruta al archivo del modelo entrenado.
            state_shape: Forma del estado del juego.
            n_actions: Número de acciones posibles.
            device: Dispositivo para computación.
        """   
        self.device = torch.device(device)
        self.n_actions = n_actions

        # DQN debe coincidir con la arquitectura usada en training.
        # Si tu DQN espera (rows, cols), desempaquetamos:
        rows, cols = state_shape
        self.net = DQN((rows, cols), n_actions).to(self.device)

        # Carga flexible del checkpoint
        ckpt = torch.load(model_path, map_location=self.device)
        if isinstance(ckpt, dict) and "policy" in ckpt:
            # Guardaste un dict con varias cosas (policy/target/opt/epsilon/etc.)
            self.net.load_state_dict(ckpt["policy"])
        else:
            # Guardaste sólo el state_dict de la policy
            self.net.load_state_dict(ckpt)

        self.net.eval()

    def play(self, state, valid_actions): 
        """
        Selecciona la mejor acción según el modelo entrenado.
        
        Args:
            state: Estado actual del juego.
            valid_actions: Lista de acciones válidas.
            
        Returns:
            Índice de la mejor acción según el modelo.
        """
        with torch.no_grad():
            s_t = self.preprocess(state)    
            q = self.net(s_t)                                     # [1, n_actions]

            # Enmascarar acciones inválidas
            mask = torch.full_like(q, -1e9)                       # [1, n_actions]
            mask[:, valid_actions] = q[:, valid_actions]

            action = int(torch.argmax(mask, dim=1).item())
            return action

