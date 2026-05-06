"""
Control Systems for AI - Additional Models
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

class AIControllerModels:
    """
    Additional AI control system models
    """
    
    def __init__(self):
        pass
    
    def adaptive_controller(self, t_final=10, n_points=1000):
        """
        Model-Reference Adaptive Control (MRAC) for AI
        """
        t = np.linspace(0, t_final, n_points)
        
        # Reference model
        ref_num = [1]
        ref_den = [1, 2, 1]
        ref_sys = signal.TransferFunction(ref_num, ref_den)
        _, ref_response = signal.step(ref_sys, T=t)
        
        # Adaptive plant
        adaptive_response = ref_response + 0.1 * np.exp(-0.5 * t) * np.sin(2 * np.pi * t)
        
        return t, ref_response, adaptive_response
    
    def fuzzy_controller(self, error_values):
        """
        Simple fuzzy logic controller membership functions
        """
        # Membership functions
        NB = np.exp(-((error_values + 1) / 0.3)**2)  # Negative Big
        NS = np.exp(-((error_values + 0.5) / 0.2)**2)  # Negative Small
        Z = np.exp(-(error_values / 0.2)**2)  # Zero
        PS = np.exp(-((error_values - 0.5) / 0.2)**2)  # Positive Small
        PB = np.exp(-((error_values - 1) / 0.3)**2)  # Positive Big
        
        return NB, NS, Z, PS, PB


def main():
    """Demo control systems"""
    controller = AIControllerModels()
    
    # Test fuzzy logic
    error = np.linspace(-2, 2, 100)
    NB, NS, Z, PS, PB = controller.fuzzy_controller(error)
    
    plt.figure(figsize=(10, 5))
    plt.plot(error, NB, 'r-', label='NB', linewidth=2)
    plt.plot(error, NS, 'g-', label='NS', linewidth=2)
    plt.plot(error, Z, 'b-', label='Z', linewidth=2)
    plt.plot(error, PS, 'orange', label='PS', linewidth=2)
    plt.plot(error, PB, 'purple', label='PB', linewidth=2)
    plt.xlabel('Error')
    plt.ylabel('Membership Degree')
    plt.title('Fuzzy Logic Controller Membership Functions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('plots/fuzzy_controller.png', dpi=150)
    plt.show()

if __name__ == "__main__":
    main()
