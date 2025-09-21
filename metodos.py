
from principal import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from typing import Tuple
import random
from agentes import Agent
from collections import namedtuple



# ----------------- Bloque residual simple -----------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.rb1_conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.rb1_bn1   = nn.BatchNorm2d(out_ch)
        self.rb1_conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False)
        self.rb1_bn2   = nn.BatchNorm2d(out_ch)

        self.need_proj = (in_ch != out_ch)
        if self.need_proj:
            self.rb2_proj    = nn.Conv2d(in_ch, out_ch, 1, bias=False)
            self.rb2_proj_bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        identity = x
        out = F.relu(self.rb1_bn1(self.rb1_conv1(x)))
        out = self.rb1_bn2(self.rb1_conv2(out))
        if self.need_proj:
            identity = self.rb2_proj_bn(self.rb2_proj(identity))
        out = F.relu(out + identity)
        return out

# ----------------- Arquitectura residual (ResDQN) -----------------
class ResDQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        h, w = input_shape  # p.ej. (6, 7)
        in_ch = 1           # ajustá si usás 2/3 canales
        mid = 64

        self.conv1 = nn.Conv2d(in_ch, mid, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(mid)

        # dos bloques residuales (nombres compatibles con el checkpoint)
        self.rb1_conv1 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.rb1_bn1   = nn.BatchNorm2d(mid)
        self.rb1_conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.rb1_bn2   = nn.BatchNorm2d(mid)

        self.rb2_conv1 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.rb2_bn1   = nn.BatchNorm2d(mid)
        self.rb2_conv2 = nn.Conv2d(mid, mid, 3, padding=1, bias=False)
        self.rb2_bn2   = nn.BatchNorm2d(mid)

        # proyección opcional en el ckpt (si estuviera guardada)
        self.rb2_proj    = nn.Conv2d(mid, mid, 1, bias=False)
        self.rb2_proj_bn = nn.BatchNorm2d(mid)

        flat_dim = mid * h * w

        # "cabeza por columnas" (coincidir nombres)
        self.col_head1   = nn.Linear(flat_dim, 256, bias=True)
        self.col_ln1   = nn.LayerNorm(256)
        self.col_head_out= nn.Linear(256, n_actions, bias=True)

        # algunos checkpoints guardan un sesgo por columna
        self.col_bias = nn.Parameter(torch.zeros(n_actions))
        nn.init.uniform_(self.col_head_out.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.col_head_out.bias)
        nn.init.zeros_(self.col_bias)  # ya estaba en cero, lo explicitamos


    def forward(self, x):
        # x: [B, in_ch=1, H, W]
        x = F.relu(self.bn1(self.conv1(x)))

        # bloque 1
        y = F.relu(self.rb1_bn1(self.rb1_conv1(x)))
        y = self.rb1_bn2(self.rb1_conv2(y))
        x = F.relu(x + y)

        # bloque 2
        y = F.relu(self.rb2_bn1(self.rb2_conv1(x)))
        y = self.rb2_bn2(self.rb2_conv2(y))
        # si existiera proyección diferente, usarla; aquí dimensiona igual
        x = F.relu(x + y)

        x = torch.flatten(x, 1)
        x = self.col_head1(x)
        # BatchNorm1d requiere B>1; en eval con B=1 podemos desactivarla
        #if self.training and x.size(0) > 1:
        #    x = self.col_bn1(x)
        
        x = self.col_ln1(x)
        x = F.relu(x)
        q = self.col_head_out(x) + self.col_bias
        return q

# ----------------- DQN "clásico" que ya tenías -----------------
class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super().__init__()
        h, w = input_shape
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * h * w, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, n_actions),
        )

    def forward(self, x):
        return self.head(self.features(x))

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
        #self.learningRate = lr
        self.target_update_every = target_update_every
        self.memory = []  # Memoria de experiencias
        self.memory_size = memory_size
        self.step_count = 0  # Contador de pasos para actualizar la red objetivo
        self.policy_net = ResDQN(state_shape, n_actions).to(device)
        self.target_net = ResDQN(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # La red objetivo no se entrena
        #self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        #self.loss_fn = nn.MSELoss()
        self.learningRate = 5e-4  # o 1e-4 si aún ves picos
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learningRate)
        self.loss_fn = nn.SmoothL1Loss(beta=1.0)  # Huber
        self.train_steps = 0
        self.grad_steps = 0
        self.Transition = namedtuple('Transition', ('s', 'a', 'r', 's_next', 'done'))

    def _mask_q_next(self, q_next_online: torch.Tensor, s_next_batch: torch.Tensor) -> torch.Tensor:
        # s_next_batch: [B, 1, H, W] → columnas libres en top row
        top = s_next_batch[:, 0, 0, :]         # [B, W]
        valid = (top == 0)                     # bool [B, W]
        q_next_masked = q_next_online.masked_fill(~valid, -1e9)  # [B, A] con A==W
        return q_next_masked


    def preprocess(self, state):
        """
        Convierte el estado del juego a un tensor de PyTorch.
        
        Args:
            state: Estado del juego.
            
        Returns:
            Tensor de PyTorch con el estado aplanado.
        """
        board = state.board
        arr = np.array(board, dtype=np.float32)        # (rows, cols)
        arr = np.where(arr == 2, -1.0, arr)
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
        #with torch.no_grad():
         #   s_t = self.preprocess(state)        # [rows, cols]
            ###q_values = self.policy_net(s_t)     # [1, n_actions]: asigna proba a cada posibilidad

            self.policy_net.eval()
        with torch.no_grad():
            s_t = self.preprocess(state)
            q_values = self.policy_net(s_t)
            self.policy_net.train()

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
        s_t = self.preprocess(s).squeeze(0).squeeze(0).to(self.device)         # [1, H, W] -> [H, W]
        s_next_t = self.preprocess(s_next).squeeze(0).squeeze(0).to(self.device)
        self.memory.append(self.Transition(s_t, a, r, s_next_t, done))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)  # descartar la más antigua

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
        # Cada s se guardó como [H, W] o [1, H, W]; lo pasamos a [B, 1, H, W]
        s_batch      = torch.stack([b.s.squeeze(0) if b.s.dim()==3 else b.s for b in batch], dim=0).unsqueeze(1).to(self.device)
        s_next_batch = torch.stack([b.s_next.squeeze(0) if b.s_next.dim()==3 else b.s_next for b in batch], dim=0).unsqueeze(1).to(self.device)
        a_batch      = torch.tensor([b.a for b in batch], dtype=torch.long, device=self.device)
        r_batch      = torch.tensor([b.r for b in batch], dtype=torch.float32, device=self.device)
        done_batch   = torch.tensor([b.done for b in batch], dtype=torch.bool, device=self.device)

        # Q(s,a) con gradiente
        q_all = self.policy_net(s_batch)                      # [B, A]
        q_sa  = q_all.gather(1, a_batch.unsqueeze(1)).squeeze(1)  # [B]

        # Target sin gradiente
        with torch.no_grad():
            q_next_online = self.policy_net(s_next_batch)     # [B, A]
            q_next_online_masked = self._mask_q_next(q_next_online, s_next_batch)
            next_acts = q_next_online_masked.argmax(dim=1)    # [B]

            q_next_target_all = self.target_net(s_next_batch) # [B, A]
            q_next_target = q_next_target_all.gather(1, next_acts.unsqueeze(1)).squeeze(1)  # [B]

            target = r_batch + (~done_batch).float() * self.gamma * q_next_target  # [B]

        # Loss y paso de optimización (con gradiente)
        loss = self.loss_fn(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

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

class TrainedAgent:
    def __init__(self, model_path, state_shape, n_actions, device, arch: str | None = None):
        self.device = device
        self.name = "Agente nuestro"
        # weights_only=True: evita el warning y es más seguro si guardaste state_dict
        ckpt = torch.load(model_path, map_location=self.device, weights_only=True)

        # soporta tanto {"state_dict": ...} como state_dict plano
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            ckpt_arch = ckpt.get("arch")
        else:
            state_dict = ckpt
            ckpt_arch = None

        # prioridad: flag explícita -> arch en ckpt -> autodetección por claves
        arch_name = arch or ckpt_arch
        if arch_name is None:
            ks = list(state_dict.keys())
            looks_resnet = any(k.startswith(("rb1_", "rb2_")) for k in ks) or any("col_head" in k for k in ks)
            arch_name = "resdqn" if looks_resnet else "dqn"

        if arch_name.lower() in ("res", "resdqn", "resnet"):
            self.net = ResDQN(state_shape, n_actions).to(self.device)
        else:
            self.net = DQN(state_shape, n_actions).to(self.device)

        self.net.load_state_dict(state_dict)
        self.net.eval()


    def _preprocess_single(self, state, device):
        arr = np.array(state.board, dtype=np.float32)   # (H, W)
        tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        return tensor.to(device)


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
            s_t = self._preprocess_single(state,self.device)    
            q = self.net(s_t)                                     # [1, n_actions]

            # Enmascarar acciones inválidas
            mask = torch.full_like(q, -1e9)                       # [1, n_actions]
            mask[:, valid_actions] = q[:, valid_actions]

            action = int(torch.argmax(mask, dim=1).item())
            return action

