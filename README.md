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

2. **Crear y activar entorno virtual** (recomendado):
```bash
python -m venv env
env\Scripts\activate  # Windows
# o
source env/bin/activate  # Linux/Mac
```

3. **Instalar dependencias**:
```bash
pip install torch numpy
```

4. **Verificar instalación**:
```bash
python -c "import torch; import numpy; print('✓ Todo instalado correctamente')"
```

## 🎮 Guía de Uso Completa

### 1️⃣ Jugar una Partida Simple

#### Ejemplo básico - Random vs Random:
```bash
python main.py --agent1 random --agent2 random
```

#### Humano vs Agente Defensor:
```bash
python main.py --agent1 human --agent2 defender -v
```

#### Random vs Defensor (sin mostrar tablero):
```bash
python main.py --agent1 random --agent2 defender
```

**Opciones disponibles**:
- `--rows, -r`: Número de filas del tablero (default: 6)
- `--cols, -c`: Número de columnas del tablero (default: 7)
- `--agent1, -a1`: Tipo de agente 1 (`random`, `defender`, `human`, `trained`)
- `--agent2, -a2`: Tipo de agente 2 (`random`, `defender`, `human`, `trained`)
- `--model1`: Ruta del modelo `.pth` para agente 1 (si es `trained`)
- `--model2`: Ruta del modelo `.pth` para agente 2 (si es `trained`)
- `--device`: Dispositivo (`cpu`, `cuda`, `auto`)
- `--verbose, -v`: Mostrar tablero en cada turno

---

### 2️⃣ Entrenar un Agente DQN

#### Entrenamiento básico (200 episodios vs Random):
```bash
python entrenar.py -n 200 -ot random -v
```

#### Entrenamiento largo (1000 episodios vs Random):
```bash
python entrenar.py -n 1000 -ot random -v
```

#### Entrenamiento vs Defensor (más desafiante):
```bash
python entrenar.py -n 2000 -ot defender -v
```

#### Entrenamiento avanzado con parámetros personalizados:
```bash
python entrenar.py -n 5000 -ot random -g 0.99 -es 1.0 -em 0.1 -ed 0.995 -a 0.001 -bs 128 -ms 1000 -ue 100 -v
```

**Parámetros de entrenamiento**:
- `-n, --episodes`: Número de episodios (default: 1000)
- `-ot, --opponent_type`: Tipo de oponente (`random`, `defender`) (default: defender)
- `-g, --gamma`: Factor de descuento (default: 0.99)
- `-es, --epsilon_start`: Epsilon inicial para exploración (default: 1.0)
- `-em, --epsilon_min`: Epsilon mínimo (default: 0.1)
- `-ed, --epsilon_decay`: Tasa de decaimiento de epsilon (default: 0.995)
- `-a, --alpha`: Tasa de aprendizaje (default: 0.001)
- `-bs, --batch_size`: Tamaño del batch (default: 128)
- `-ms, --memory_size`: Tamaño de la memoria de replay (default: 1000)
- `-ue, --target_update_every`: Frecuencia de actualización de red objetivo (default: 100)
- `-of, --opponent_model_path`: Ruta `.pth` para entrenar contra otro modelo entrenado
- `-v, --verbose`: Mostrar progreso cada 100 episodios

**Resultado del entrenamiento**:
- Se genera un archivo `.pth` con el modelo entrenado
- Nombre formato: `trained_model_vs_[oponente]_[params].pth`
- Ejemplo: `trained_model_vs_RandomAgent_1000_0.99_1.0_0.1_0.9950.001_128_1000_100.pth`

---

### 3️⃣ Evaluar un Agente Entrenado

#### Evaluar vs Random (100 partidas):
```bash
python eval.py --model trained_model_vs_RandomAgent_200_0.99_1.0_0.1_0.9950.001_128_1000_100.pth --opponent random --games 100
```

#### Evaluar vs Defensor (50 partidas):
```bash
python eval.py --model trained_model_vs_RandomAgent_200_0.99_1.0_0.1_0.9950.001_128_1000_100.pth --opponent defender --games 50
```

#### Evaluar vs otro modelo entrenado:
```bash
python eval.py --model modelo1.pth --opponent trained --opponent_model modelo2.pth --games 100
```

**Parámetros de evaluación**:
- `--model`: Ruta del modelo `.pth` a evaluar (obligatorio)
- `--opponent`: Tipo de oponente (`random`, `defender`, `trained`)
- `--opponent_model`: Ruta del modelo oponente (si opponent=`trained`)
- `--games`: Número de partidas a jugar (default: 100)
- `--rows`: Filas del tablero (default: 6)
- `--cols`: Columnas del tablero (default: 7)
- `--device`: Dispositivo (`cpu`, `cuda`, `auto`)
- `-v, --verbose`: Mostrar cada partida (lento)

**Resultado de la evaluación**:
```
Evaluando: Agente nuestro vs. Random | Juegos: 100 | Device: cpu
Resultados -> A gana: 70 (70.0%) | B gana: 30 (30.0%) | Empates: 0 (0.0%)
```

---

### 4️⃣ Jugar contra un Agente Entrenado

#### Tú (humano) vs Agente entrenado:
```bash
python main.py --agent1 human --agent2 trained --model2 trained_model_vs_RandomAgent_200_0.99_1.0_0.1_0.9950.001_128_1000_100.pth -v
```

#### Agente entrenado vs Defensor:
```bash
python main.py --agent1 trained --model1 trained_model_vs_RandomAgent_200_0.99_1.0_0.1_0.9950.001_128_1000_100.pth --agent2 defender -v
```

---

## 📋 Flujo de Trabajo Recomendado

### Para entrenar y evaluar un nuevo agente:

1. **Entrenar el agente** (ajusta el número de episodios según tu tiempo):
```bash
python entrenar.py -n 500 -ot random -v
```

2. **Copiar el nombre del modelo generado** (aparece en la salida):
```
trained_model_vs_RandomAgent_500_0.99_1.0_0.1_0.9950.001_128_1000_100.pth
```

3. **Evaluar el rendimiento vs Random**:
```bash
python eval.py --model trained_model_vs_RandomAgent_500_0.99_1.0_0.1_0.9950.001_128_1000_100.pth --opponent random --games 100
```

4. **Evaluar el rendimiento vs Defensor**:
```bash
python eval.py --model trained_model_vs_RandomAgent_500_0.99_1.0_0.1_0.9950.001_128_1000_100.pth --opponent defender --games 100
```

5. **Jugar tú mismo contra el agente**:
```bash
python main.py --agent1 human --agent2 trained --model2 trained_model_vs_RandomAgent_500_0.99_1.0_0.1_0.9950.001_128_1000_100.pth -v
```

---

## 🎯 Ejemplos de Comandos Rápidos

### Comandos Copy-Paste Listos

```bash
# 1. Entrenar agente rápido (200 episodios, ~2-3 minutos)
python entrenar.py -n 200 -ot random -v

# 2. Entrenar agente medio (1000 episodios, ~10-15 minutos)
python entrenar.py -n 1000 -ot random -v

# 3. Entrenar agente fuerte (5000 episodios, ~45-60 minutos)
python entrenar.py -n 5000 -ot defender -v

# 4. Evaluar cualquier modelo (reemplaza NOMBRE_MODELO.pth)
python eval.py --model NOMBRE_MODELO.pth --opponent random --games 100

# 5. Jugar contra el modelo (reemplaza NOMBRE_MODELO.pth)
python main.py --agent1 human --agent2 trained --model2 NOMBRE_MODELO.pth -v

# 6. Ver partida: Random vs Defensor
python main.py --agent1 random --agent2 defender -v

# 7. Listar todos los modelos entrenados
dir *.pth  # Windows
ls *.pth   # Linux/Mac
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