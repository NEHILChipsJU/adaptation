# Adaption in Neuromorphic Chips

This repository presents an exploration of **conceptors** and the **Conceptor Control Loop (CCL)**, introduced in Pourcel et al. (2024), to enhance the adaptivity of **Reservoir Computing (RC)** systems and to study their potential for **neuromorphic implementations**.  

The work aims to demonstrate how conceptors can transcend static projections by adapting online through the CCL, allowing RC systems to dynamically adjust to new temporal patterns, an essential property for robust neuromorphic computation.

---

## Task 1: Sinusoidal Interpolation

The RC was trained with **two sine waves** of equal amplitude but different periods.  
Each sine wave defines a distinct internal reservoir state, represented by two conceptors: **C₁** and **C₂**.  

To interpolate between them, the following definition is used:

$$
C_{interp} = \lambda C_{1} + (1-\lambda)C_{2}, \quad \lambda \in [0,1]
$$

The objective is to generate an output corresponding to the interpolated conceptor $C_{interp}$.

---
## Task 1: Network degradation

---
## Repository Structure
- **`utils`**

	- **`rrnn_utils.py`**  
  Contains functions for training the reservoir, computing internal states, and calculating conceptors.  

	- **`utils.py`**  
  Provides functions for visualization and presentation of results.  

- **`sw_interpolation.py`**  
  Implements interpolation using a *fixed* interpolated conceptor $C_{\text{interp}}$.  

- **`sw_interpolation_CCL.py`**  
  Implements interpolation using the **Conceptor Control Loop (CCL)**, which adapts conceptors online during internal state computation.  

---

## Results
### Task 1
By scanning the interpolation parameter $\lambda \in [0,1]$, the following observations were obtained:  

- **constant interpolated conceptor** is able to perform interpolation only under limited conditions.  
- However, when the **Conceptor Control Loop (CCL)** is applied, the range of conditions where interpolation can be successfully achieved is significantly extended.

These results are consistent with the findings of Pourcel et al. (2024), emphasizing the potential of adaptive mechanisms for neuromorphic hardware applications.

### Task 2
---

## Reference

Pourcel, G., Goldmann, M., Fischer, I., & Soriano, M. C. (2024). *Adaptive control of recurrent neural networks using conceptors.* **Chaos, 34**(10), 103127. https://doi.org/10.1063/5.0211692  
