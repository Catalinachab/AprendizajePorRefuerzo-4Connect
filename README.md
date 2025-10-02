# Connect 4 con Aprendizaje por Refuerzo 🔴🔵

Un proyecto de inteligencia artificial que implementa el juego **Connect 4** (Conecta 4) utilizando **Deep Q-Learning** para entrenar agentes que aprenden a jugar de forma autónoma.

## 🎯 Descripción del Proyecto

Este proyecto implementa diferentes tipos de agentes para jugar Connect 4:
- **Agente Random**: Juega movimientos aleatorios
- **Agente Defensor**: Intenta bloquear las jugadas ganadoras del oponente
- **Agente Humano**: Permite a un humano jugar contra la máquina
- **Agente DQN**: Utiliza Deep Q-Learning para aprender estrategias óptimas

## 🚀 Características

- **Entorno de juego personalizable**: Tableros de diferentes tamaños (por defecto 6x7)
- **Múltiples algoritmos de IA**: Desde estrategias simples hasta aprendizaje profundo
- **Entrenamiento automático**: Sistema de entrenamiento con parámetros configurables
- **Evaluación de rendimiento**: Herramientas para evaluar y comparar agentes
- **Interfaz de línea de comandos**: Fácil interacción y configuración

## 📁 Estructura del Proyecto

```
├── agentes.py              # Definición de diferentes tipos de agentes
├── connect4.py             # Clase principal del juego Connect 4
├── principal.py            # Entorno y estado del juego
├── metodos.py              # Implementación de algoritmos de RL (DQN)
├── entrenar.py             # Script para entrenar agentes DQN
├── main.py                 # Interfaz principal para jugar partidas
├── eval.py                 # Evaluación y comparación de agentes
├── jugar_humano_contra_defensor.py  # Partida humano vs agente
├── utils.py                # Funciones utilitarias
└── trained_model_*.pth     # Modelos entrenados guardados
```

## 🛠️ Instalación

1. **Clonar el repositorio**:
```bash
git clone https://github.com/Catalinachab/AprendizajePorRefuerzo-4Connect.git
cd AprendizajePorRefuerzo-4Connect
```

2. **Instalar dependencias**:
```bash
pip install torch numpy
```

## 🎮 Uso

### Jugar una Partida

Para jugar una partida entre dos agentes:

```bash
python main.py --agent1 human --agent2 defender
```

**Opciones disponibles**:
- `--rows, -r`: Número de filas del tablero (default: 6)
- `--cols, -c`: Número de columnas del tablero (default: 7)
- `--agent1, -a1`: Tipo de agente 1 (random, defender, human, trained)
- `--agent2, -a2`: Tipo de agente 2 (random, defender, human, trained)
- `--model1`: Ruta del modelo para agente entrenado 1
- `--model2`: Ruta del modelo para agente entrenado 2
- `--device`: Dispositivo de cómputo (cpu, cuda)

### Entrenar un Agente DQN

Para entrenar un nuevo agente:

```bash
python entrenar.py --episodes 1000 --opponent random
```

**Parámetros de entrenamiento**:
- `--episodes`: Número de episodios de entrenamiento
- `--gamma`: Factor de descuento (default: 0.99)
- `--epsilon_start`: Epsilon inicial para exploración (default: 1.0)
- `--epsilon_min`: Epsilon mínimo (default: 0.1)
- `--epsilon_decay`: Tasa de decaimiento de epsilon (default: 0.995)
- `--alpha`: Tasa de aprendizaje (default: 0.001)
- `--batch_size`: Tamaño del batch (default: 64)
- `--opponent`: Tipo de oponente (random, defender, None para self-play)

### Evaluar Agentes

Para evaluar el rendimiento de un agente entrenado:

```bash
python eval.py --model path/to/model.pth --opponent random --games 100
```

### Ejemplos de Uso

1. **Humano vs Agente Defensor**:
```bash
python main.py --agent1 human --agent2 defender
```

2. **Entrenar contra agente aleatorio**:
```bash
python entrenar.py --episodes 2000 --opponent random --alpha 0.001
```

3. **Agente entrenado vs Agente defensor**:
```bash
python main.py --agent1 trained --model1 trained_model_vs_RandomAgent_2000_0.99_1.0_0.1_0.9950.001_128_1000_100.pth --agent2 defender
```

## 🧠 Algoritmo de Aprendizaje

El proyecto implementa **Deep Q-Learning (DQN)** con las siguientes características:

- **Red neuronal profunda** para aproximar la función Q
- **Experience Replay** para mejorar la estabilidad del entrenamiento
- **Target Network** para reducir la correlación temporal
- **Epsilon-greedy exploration** para balancear exploración y explotación

### Arquitectura de la Red Neuronal

- Entrada: Estado del tablero (6x7 por defecto)
- Capas ocultas: Fully connected layers con activación ReLU
- Salida: Q-values para cada acción posible (7 acciones por defecto)

## 📊 Resultados

Los agentes entrenados muestran mejoras significativas en el rendimiento:

- **Agente Random**: ~25% de victorias (línea base)
- **Agente Defensor**: ~40-50% de victorias contra random
- **Agente DQN**: >70% de victorias contra random después del entrenamiento

---

**Autores**: Juan Ignacio Castore, Catalina Chab, Catalina Brusco  
**Materia**: Inteligencia Artificial y Neurociencias