# Tensaur - Quadruped Robot with Tensegrity Spine

This folder contains scripts to generate and run a tensegrity-based quadruped robot simulation.

## Requirements

At least Python 3.11 and the following packages:

- `dm_control`
- `mujoco` 
- `jax[cuda12]`
- `numpy`

## Quick Start

### Installation

To install the tensaur environment:

```bash
cd <YOUR_PATH>/tensaur
pip install -e .
```

### Generate the Robot Morphology

#### Pleurobot with Tensegrity Spine
To generate the Pleurobot with a tensegrity spine, run the following script:

```bash
python src/tensegrity_playground/envs/tensaur/generate_pleurobot_tendon.py
```

The generated XML file will be saved at:

```
.../src/tensegrity_playground/envs/tensaur/xmls/PleurobotII/pleurobot_tensegrity_0.xml
```

---

### Running the Simulation

To start the simulation for the Pleurobot with a tensegrity spine, use the following commands:

1. **Run the radius environment:**

   ```bash
   python scripts/train_radius.py playground=PleurobotRadius
   ```

   The corresponding environment file is located at:

   ```
   .../src/tensegrity_playground/envs/tensaur/radius.py
   ```

2. **Run the walking environment:**

   ```bash
   python scripts/train_run.py playground=PleurobotRun
   ```

   The corresponding environment file is located at:

   ```
   .../src/tensegrity_playground/envs/tensaur/run.py
   ```

---

## Files under src/tensegrity_playground/envs/tensaur/

- **`generate_morphology.py`** - Creates the robot morphology (structure, joints, tendons)
- **`generate_pleurobot_tendon.py`** - Creates the Pleurobot morphology with a tensegrity spine
- **`radius.py`** - Environment file for the Pleurobot radius simulation
- **`run.py`** - Environment file for the Pleurobot running simulation
- **[`xmls`](xmls )** - Generated MuJoCo XML files, including:
  - `PleurobotII/pleurobot_tensegrity_0.xml`: Pleurobot with tensegrity spine
    - `scene_tensegrity_quadruped.xml`: Go1-like tensegrity quadruped robot

---

## Customizing the Robot

### Key Parameters in `generate_morphology.py`

The `DEFAULT_CONFIG` dictionary contains all morphology parameters:

**Spine Structure:**
- `num_segments`: Number of vertebrae (default: 8)
- `segment_spacing`: Distance between vertebrae (default: 0.06m)
- `alpha`/`beta`: Tendon angles for structural stability
- `lateral_pretension`/`diagonal_pretension`: Tendon tension (0.0-1.0)

**Leg Dimensions:**
- `hip_roll_length`: Upper leg segment length
- `hip_pitch_length`: Mid leg segment length  
- `shin_length`: Lower leg segment length
- `foot_radius`: Foot contact sphere size

**Physics:**
- `timestep`: Simulation timestep (default: 0.001s)
- `integrator`: Physics integrator ("Euler" or "RK4")
