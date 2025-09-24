# Adaptation in Neuromorphic Chips

This repository is part of the EU-funded **NEHIL project** ([NEHIL.eu](https://www.nehil.eu/), Project ID [101194363](https://cordis.europa.eu/project/id/101194363)), which addresses the growing need for more efficient and powerful computing systems by developing new neuromorphic architectures.  

Within this context, the repository presents an exploration of **conceptors** and the **Conceptor Control Loop (CCL)**, as introduced in Pourcel et al. (2024), to enhance the adaptivity of **Reservoir Computing (RC)** systems and to assess their potential for **neuromorphic implementations**.  

The study demonstrates how conceptors can go beyond static projections by adapting online through the CCL, enabling RC systems to dynamically adjust to new temporal patterns, an essential property for robust neuromorphic computation.

The reservoir is implemented using an Echo State Network (ESN).

---

## Task 1: Sinusoidal Interpolation

The ESN was trained with **two sine waves** of equal amplitude but different periods.  
Each sine wave defines a distinct internal reservoir state, represented by two conceptors: **C₁** and **C₂**.  

To interpolate between them, the following definition is used:

$$
C_{interp} = \lambda C_{1} + (1-\lambda)C_{2}, \quad \lambda \in [0,1]
$$

The objective is to generate an output corresponding to the interpolated conceptor $C_{interp}$.

---
## Task 2: Network degradation
For this task, the ESN was trained on a single sinusoidal signal and the then was **progressively degraded** by disabling different subsets of neurons. The goal was to evaluate the **robustness** of the ESN, to this end, two approaches are considered: applying a constant conceptor (corresponding to the trained sinusoidal signal) and applying the Conceptor Control Loop (CCL). Their performance is compared in order to demonstrate that the adaptive nature of the CCL provides a clear advantage in maintaining network robustness.

This experiment is directly related to the notion of **"graceful degradation"**, a property shared by both artificial and biological neural networks, where performance decreases gradually as more units are lost (McClelland et al. (1986)). 

---
## Repository Structure
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

- The **constant interpolated conceptor** can perform interpolation only under limited conditions. As illustrated in the figures below, the greater the difference between the periods of the sine waves used to train the ESN, the more difficult it becomes for the constant conceptor to achieve successful interpolation.
  
  <img src="figures/Interpolation/y_lamda_scan_C.png" alt="Interpolation with constant conceptor" width="360"/> <img src="figures/Interpolation/Instanteneous_Periode_C.png" alt="Interpolation with constant conceptor" width="400"/> 
  
- However, when the **Conceptor Control Loop (CCL)** is applied, the range of conditions where interpolation can be successfully achieved is significantly extended.

  <img src="figures/Interpolation/y_lamda_scan_CCL.png" alt="Interpolation with CCL" width="360"/> <img src="figures/Interpolation/Instanteneous_Periode_CCL.png" alt="Interpolation with CCL" width="400"/> 

These results are consistent with the findings of Pourcel et al. (2024), emphasizing the potential of adaptive mechanisms for neuromorphic hardware applications.

### Task 2
A **qualitative assessment** was conducted after the removal of K neurons. For each increment of K, 70 trials were performed with randomly removed neurons. The network was considered to fail when the variance of the principal component of the degraded network ($\sigma^2_{degraded}$) dropped below 10% of the variance observed in the non-degraded network ($\sigma^2$).

$$
\sigma^2_{degraded}< 0.1 \cdot \sigma^2
$$

The results show that, when using either a constant conceptor or the CCL, the failure rate of the network with 10.7% of its neurons disabled (60 neurons) remains below 20%. Furthermore, with the CCL, the failure rate can be reduced to about 5% under the same level of degradation and maintained below 20% even for higher levels of degradation. These findings highlight the effectiveness of the Conceptor Control Loop in enhancing the robustness of Reservoir Computing systems.
<div align="center">
<img src="figures/Network degradation/qualitative_rep70_th0.1_N560_reg1.0_a25_beta0.4_nu2.5e-05_T20_sr1.6_alpha0.75_scaling0.9_bs0.4.png" alt="Qualitative Evaluation, network degradation" width="360"/>
</div>

---

## References

- Pourcel, G., Goldmann, M., Fischer, I., & Soriano, M. C. (2024). *Adaptive control of recurrent neural networks using conceptors.* **Chaos, 34**(10), 103127. https://doi.org/10.1063/5.0211692  
- McClelland, J. L., Rumelhart, D. E., & PDP Research Group. (1986).  
*Parallel Distributed Processing, Volume 2: Explorations in the Microstructure of Cognition: Psychological and Biological Models.*  
Cambridge, MA: MIT Press.
