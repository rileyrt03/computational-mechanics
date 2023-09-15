---
jupytext:
  formats: notebooks//ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
```

# Computational Mechanics Project #01 - Heat Transfer in Forensic Science

We can use our current skillset for a macabre application. We can predict the time of death based upon the current temperature and change in temperature of a corpse. 

Forensic scientists use Newton's law of cooling to determine the time elapsed since the loss of life, 

$\frac{dT}{dt} = -K(T-T_a)$,

where $T$ is the current temperature, $T_a$ is the ambient temperature, $t$ is the elapsed time in hours, and $K$ is an empirical constant. 

Suppose the temperature of the corpse is 85$^o$F at 11:00 am. Then, 45
min later the temperature is 80$^{o}$F. 

Assume ambient temperature is a constant 65$^{o}$F.

1. Use Python to calculate $K$ using a finite difference approximation, $\frac{dT}{dt} \approx \frac{T(t+\Delta t)-T(t)}{\Delta t}$.

```{code-cell} ipython3
Ta = 65
Tt = 85
TDeltaTPlust = 80
Delta_t = 0.75 #in hours

dT_dt = (TDeltaTPlust - Tt)/Delta_t
K = -(dT_dt)/(Tt - Ta)
print(K)
```

2. Change your work from problem 1 to create a function that accepts the temperature at two times, ambient temperature, and the time elapsed to return $K$.

```{code-cell} ipython3
def calculate_K(T1, T2, Ta, TimeElapsed):
    K = -((T2 - T1) / (TimeElapsed * (T1 - Ta)))
    return K
```

```{code-cell} ipython3
calculate_K(85, 80, 65, 0.75)
```

3. A first-order thermal system has the following analytical solution, 

    $T(t) =T_a+(T(0)-T_a)e^{-Kt}$

    where $T(0)$ is the temperature of the corpse at t=0 hours i.e. at the time of discovery and $T_a$ is a constant ambient temperature. 

    a. Show that an Euler integration converges to the analytical solution as the time step is decreased. Use the constant $K$ derived above and the initial temperature, T(0) = 85$^o$F. 

    b. What is the final temperature as t$\rightarrow\infty$?
    
    c. At what time was the corpse 98.6$^{o}$F? i.e. what was the time of death?

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

Ta = 65
T0 = 85
K = 0.3333333333333333

t_final = 10
num_steps = [10, 100, 1000]
analytical_solution = []
euler_solution = []

def analytical_solution_at_t(t):
    return Ta + (T0 - Ta) * np.exp(-K * t)

times = np.linspace(0, t_final, num_steps[-1] + 1)

for num_steps_per_dt in num_steps:
    dt = t_final / num_steps_per_dt
    euler_result = [T0]
    
    for i in range(1, len(times)):
        cooling_term = K * (euler_result[-1] - Ta)
        euler_result.append(euler_result[-1] - dt * cooling_term)

    analytical_solution.append([analytical_solution_at_t(t) for t in times])
    euler_solution.append(euler_result)

plt.figure(figsize=(12, 6))
for i, num_steps_per_dt in enumerate(num_steps):
    plt.plot(times, euler_solution[i], linestyle='--', marker='o', markersize=3, label=f'Euler (dt={t_final/num_steps_per_dt:.2f} hours)')

plt.xlabel('Time (hours)')
plt.ylabel('Temperature (°F)')
plt.title('Euler Integration Convergence to Analytical Solution')
plt.legend()
plt.grid(True)
plt.show()
```

```{code-cell} ipython3
#part B
print("As t → ∞, the temperature approaches the ambient temperature (Ta). So, the final temperature as t → ∞ is 65°F (the ambient temperature).")
```

```{code-cell} ipython3
import numpy as np

# Given data
Ta = 65  # Ambient temperature in °F
T0 = 85  # Initial temperature in °F at 11:00 AM
K = 0.333333333  # Given value of K
desired_temperature = 98.6  # °F

# Calculate the time of death with error handling for denominator zero
denominator = (T0 - Ta)
if denominator == 0:
    print("The temperature will never reach the desired temperature. There is no valid time of death.")
else:
    time_of_death = -np.log((desired_temperature - Ta) / denominator) / K
    time_of_death = str(round(11.0 + time_of_death)) + ":" + str(round(((11.0 + time_of_death) % round(11.0 + time_of_death)) * 60))
    print(f"The time of death was approximately {time_of_death} AM.")
```

```{code-cell} ipython3

```
