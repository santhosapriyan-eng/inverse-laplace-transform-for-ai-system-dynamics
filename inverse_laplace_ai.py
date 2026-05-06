

### inverse_laplace_ai.py
```python
"""
Inverse Laplace Transform for AI System Dynamics
Reconstructing time-domain behavior of AI systems
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.fft import ifft
import sympy as sp
from sympy.integrals import inverse_laplace_transform
from sympy.abc import s, t
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['font.size'] = 12

class InverseLaplaceAI:
    """
    Inverse Laplace Transform analysis for AI systems
    """
    
    def __init__(self):
        """Initialize the analyzer"""
        self.s = sp.Symbol('s')
        self.t = sp.Symbol('t', positive=True)
        
    def analytical_inverse(self, F_s):
        """
        Compute analytical inverse Laplace transform
        
        Args:
            F_s: Symbolic expression in s-domain
        
        Returns:
            Symbolic expression in time-domain
        """
        return inverse_laplace_transform(F_s, self.s, self.t)
    
    def numerical_inverse(self, F, t_values, method='fourier'):
        """
        Compute numerical inverse Laplace transform
        
        Args:
            F: Function handle in s-domain
            t_values: Time points
            method: Inversion method ('fourier', 'talbot')
        
        Returns:
            Time-domain values
        """
        if method == 'fourier':
            return self._fourier_inversion(F, t_values)
        else:
            return self._talbot_inversion(F, t_values)
    
    def _fourier_inversion(self, F, t_values):
        """Fourier series based inversion"""
        n_points = len(t_values)
        s_values = 1j * np.linspace(-100, 100, 2000)
        F_values = np.array([F(s_val) for s_val in s_values])
        return np.real(ifft(F_values)[:n_points])
    
    def _talbot_inversion(self, F, t_values):
        """Talbot's method for numerical inversion"""
        N = 512
        M = 30
        results = []
        
        for t_val in t_values:
            if t_val == 0:
                results.append(0)
                continue
            
            theta = np.linspace(-M + 0.5, M - 0.5, N) * np.pi / M
            gamma = 0.5 * M / t_val
            s_vals = gamma * theta * (1j + np.tan(theta) / M)
            F_vals = np.array([F(s_val) for s_val in s_vals])
            
            integral = np.sum(F_vals * (1 + 1j * theta / (M * np.cos(theta)**2)) * np.exp(s_vals * t_val)) / M
            results.append(gamma * np.real(integral))
        
        return np.array(results)
    
    def neuron_response(self, tau=0.01, t_final=0.1, n_points=1000):
        """
        Simulate neuron activation response
        
        Neuron model as RC circuit: H(s) = 1/(τs + 1)
        
        Args:
            tau: Time constant (seconds)
            t_final: Final time (seconds)
            n_points: Number of time points
        
        Returns:
            Time and response arrays
        """
        t_values = np.linspace(0, t_final, n_points)
        
        # Analytical solution: h(t) = (1/τ) * e^(-t/τ)
        response = (1/tau) * np.exp(-t_values / tau)
        
        return t_values, response
    
    def second_order_system(self, omega_n=10, zeta=0.7, t_final=2, n_points=1000):
        """
        Second order system response (e.g., learning dynamics)
        H(s) = ω_n² / (s² + 2ζω_n s + ω_n²)
        
        Args:
            omega_n: Natural frequency
            zeta: Damping ratio
            t_final: Final time
            n_points: Number of points
        
        Returns:
            Time and step response
        """
        t_values = np.linspace(0, t_final, n_points)
        
        # Create transfer function
        num = [omega_n**2]
        den = [1, 2*zeta*omega_n, omega_n**2]
        sys = signal.TransferFunction(num, den)
        
        # Get step response
        t_response, response = signal.step(sys, T=t_values)
        
        return t_response, response
    
    def pid_controller_response(self, Kp=1, Ki=1, Kd=0.5, t_final=5, n_points=1000):
        """
        PID controller response for AI feedback systems
        H(s) = Kp + Ki/s + Kd·s
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            t_final: Final time
            n_points: Number of points
        
        Returns:
            Time and response
        """
        t_values = np.linspace(0, t_final, n_points)
        
        # Transfer function with plant G(s) = 1/(s+1)
        num_pid = [Kd, Kp, Ki]
        den_pid = [1, 0, 0]  # PID controller
        
        # Plant
        num_plant = [1]
        den_plant = [1, 1]
        
        # Closed loop: G/(1 + G*H)
        num_closed = np.convolve(num_pid, num_plant)
        den_closed = np.convolve(den_pid, den_plant) + np.convolve(num_pid, num_plant)
        
        sys = signal.TransferFunction(num_closed, den_closed)
        t_response, response = signal.step(sys, T=t_values)
        
        return t_response, response
    
    def analyze_network_dynamics(self, num_neurons=10, t_final=1, n_points=500):
        """
        Analyze recurrent neural network dynamics
        
        Args:
            num_neurons: Number of neurons in network
            t_final: Simulation time
            n_points: Number of time points
        
        Returns:
            Network response matrix
        """
        t_values = np.linspace(0, t_final, n_points)
        
        # Random connection matrix
        np.random.seed(42)
        W = np.random.randn(num_neurons, num_neurons) * 0.5
        np.fill_diagonal(W, -1)  # Self-inhibition
        
        # Initial condition
        x0 = np.random.rand(num_neurons)
        
        # Simulate dynamics: dx/dt = tanh(Wx) - x
        def network_dynamics(x, t):
            return np.tanh(W @ x) - x
        
        from scipy.integrate import odeint
        response = odeint(network_dynamics, x0, t_values)
        
        return t_values, response
    
    def stability_analysis(self, poles, zeros):
        """
        Analyze system stability from pole-zero plot
        
        Args:
            poles: List of pole locations
            zeros: List of zero locations
        """
        plt.figure(figsize=(10, 8))
        
        # Plot poles and zeros
        poles_real = [p.real for p in poles]
        poles_imag = [p.imag for p in poles]
        zeros_real = [z.real for z in zeros]
        zeros_imag = [z.imag for z in zeros]
        
        plt.plot(poles_real, poles_imag, 'rx', markersize=12, linewidth=2, label='Poles')
        plt.plot(zeros_real, zeros_imag, 'bo', markersize=8, linewidth=2, label='Zeros')
        
        # Plot unit circle for stability reference
        theta = np.linspace(0, 2*np.pi, 100)
        plt.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit Circle')
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        plt.xlabel('Real Axis')
        plt.ylabel('Imaginary Axis')
        plt.title('Pole-Zero Plot for Stability Analysis')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        
        # Determine stability
        stable = all(p.real < 0 for p in poles)
        stability_text = "STABLE" if stable else "UNSTABLE"
        color = 'green' if stable else 'red'
        plt.text(0.5, -1.2, f"System is {stability_text}", 
                fontsize=14, color=color, ha='center', transform=plt.gca().transAxes)
        
        plt.savefig('plots/stability_analysis.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return stable
    
    def plot_system_responses(self):
        """
        Plot various system responses
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. First-order response (Neuron)
        t1, y1 = self.neuron_response(tau=0.01, t_final=0.05)
        axes[0, 0].plot(t1*1000, y1, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (ms)')
        axes[0, 0].set_ylabel('Response')
        axes[0, 0].set_title('Neuron Activation Response\n(First Order)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Second-order response (Learning dynamics)
        t2, y2 = self.second_order_system(omega_n=10, zeta=0.5, t_final=2)
        axes[0, 1].plot(t2, y2, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Response')
        axes[0, 1].set_title('Learning Rate Dynamics\n(Second Order, ζ=0.5)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Critically damped
        t3, y3 = self.second_order_system(omega_n=10, zeta=1.0, t_final=2)
        axes[0, 2].plot(t3, y3, 'g-', linewidth=2)
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Response')
        axes[0, 2].set_title('Critically Damped (ζ=1.0)')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. PID controller
        t4, y4 = self.pid_controller_response(Kp=2, Ki=1, Kd=0.5, t_final=5)
        axes[1, 0].plot(t4, y4, 'purple', linewidth=2)
        axes[1, 0].axhline(y=1, color='k', linestyle='--', alpha=0.5, label='Setpoint')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Response')
        axes[1, 0].set_title('PID Controller Response\n(AI Feedback Control)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Neural network dynamics
        t5, y5 = self.analyze_network_dynamics(num_neurons=5, t_final=2)
        for i in range(min(5, y5.shape[1])):
            axes[1, 1].plot(t5, y5[:, i], linewidth=1.5, alpha=0.7, label=f'Neuron {i+1}')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Activation')
        axes[1, 1].set_title('Recurrent Neural Network Dynamics')
        axes[1, 1].legend(loc='upper right', fontsize=8)
        axes[1, 1].grid(True, alpha=0.3)
        
        # 6. Step response comparison
        tau_values = [0.005, 0.01, 0.02, 0.05]
        for tau in tau_values:
            t, y = self.neuron_response(tau=tau, t_final=0.1)
            axes[1, 2].plot(t*1000, y, linewidth=2, label=f'τ={tau*1000:.1f}ms')
        axes[1, 2].set_xlabel('Time (ms)')
        axes[1, 2].set_ylabel('Response')
        axes[1, 2].set_title('Effect of Time Constant\n(Neuron Response Speed)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/system_responses.png', dpi=150, bbox_inches='tight')
        plt.show()
    
    def symbolic_demo(self):
        """
        Demo symbolic inverse Laplace transforms
        """
        print("\n" + "="*60)
        print("SYMBOLIC INVERSE LAPLACE TRANSFORM DEMO")
        print("="*60)
        
        # Define common transfer functions
        functions = [
            ("Step Response", 1/sp.Symbol('s')),
            ("Exponential Decay", 1/(s + 2)),
            ("First Order Lag", 1/(s*(s + 1))),
            ("Second Order", 1/(s**2 + 2*s + 5)),
            ("Oscillatory", 10/(s**2 + 100)),
        ]
        
        for name, F_s in functions:
            print(f"\n{name}:")
            print(f"  F(s) = {F_s}")
            f_t = self.analytical_inverse(F_s)
            print(f"  f(t) = {f_t}")
    
    def parameter_sensitivity(self):
        """
        Analyze parameter sensitivity for second-order systems
        """
        zeta_values = [0.2, 0.5, 0.7, 1.0, 1.5, 2.0]
        
        plt.figure(figsize=(10, 6))
        
        for zeta in zeta_values:
            t, y = self.second_order_system(omega_n=10, zeta=zeta, t_final=2)
            label = f'ζ={zeta}'
            if zeta < 1:
                label += ' (Underdamped)'
            elif zeta == 1:
                label += ' (Critically Damped)'
            else:
                label += ' (Overdamped)'
            plt.plot(t, y, linewidth=2, label=label)
        
        plt.xlabel('Time (s)')
        plt.ylabel('Response')
        plt.title('Second Order System Response vs Damping Ratio')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        plt.savefig('plots/parameter_sensitivity.png', dpi=150, bbox_inches='tight')
        plt.show()


class AIControlSystem:
    """
    AI-specific control system models
    """
    
    def __init__(self):
        self.inv_laplace = InverseLaplaceAI()
    
    def reinforcement_learning_dynamics(self, alpha=0.1, gamma=0.9, t_final=100):
        """
        Model RL learning dynamics as a first-order system
        """
        t = np.arange(t_final)
        # Learning curve: value function approximation
        V = 1 - np.exp(-alpha * t)
        return t, V
    
    def gradient_descent_dynamics(self, learning_rate=0.01, t_final=100):
        """
        Model gradient descent optimization dynamics
        """
        t = np.arange(t_final)
        # Loss function decay
        loss = np.exp(-learning_rate * t)
        return t, loss
    
    def plot_ai_dynamics(self):
        """
        Plot various AI system dynamics
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # RL learning
        t_rl, V_rl = self.reinforcement_learning_dynamics()
        axes[0].plot(t_rl, V_rl, 'b-', linewidth=2)
        axes[0].set_xlabel('Episodes')
        axes[0].set_ylabel('Value Function')
        axes[0].set_title('Reinforcement Learning\nValue Convergence')
        axes[0].grid(True, alpha=0.3)
        
        # Gradient descent
        lr_values = [0.005, 0.01, 0.02, 0.05]
        for lr in lr_values:
            t, loss = self.gradient_descent_dynamics(learning_rate=lr)
            axes[1].plot(t, loss, linewidth=2, label=f'LR={lr}')
        axes[1].set_xlabel('Iterations')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Gradient Descent Convergence')
        axes[1].legend()
        axes[1].grid(True, alpha=3)
        axes[1].set_yscale('log')
        
        # Neuron firing rate
        t, rate = self.inv_laplace.neuron_response(tau=0.02, t_final=0.2)
        axes[2].plot(t*1000, rate, 'r-', linewidth=2)
        axes[2].fill_between(t*1000, 0, rate, alpha=0.3, color='red')
        axes[2].set_xlabel('Time (ms)')
        axes[2].set_ylabel('Firing Rate (Hz)')
        axes[2].set_title('Neuron Firing Rate Response')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/ai_dynamics.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """Main execution function"""
    print("\n" + "="*60)
    print("INVERSE LAPLACE TRANSFORM FOR AI SYSTEM DYNAMICS")
    print("Time-Domain Reconstruction and Analysis")
    print("="*60)
    
    # Initialize analyzer
    analyzer = InverseLaplaceAI()
    ai_control = AIControlSystem()
    
    # Symbolic demo
    analyzer.symbolic_demo()
    
    # Plot system responses
    print("\n📊 GENERATING SYSTEM RESPONSE PLOTS...")
    analyzer.plot_system_responses()
    
    # Parameter sensitivity analysis
    print("\n📈 PARAMETER SENSITIVITY ANALYSIS...")
    analyzer.parameter_sensitivity()
    
    # AI dynamics
    print("\n🤖 AI SYSTEM DYNAMICS...")
    ai_control.plot_ai_dynamics()
    
    # Stability analysis example
    print("\n🔍 STABILITY ANALYSIS...")
    poles = [complex(-1, 2), complex(-1, -2), complex(-2, 0)]
    zeros = [complex(-0.5, 0)]
    stable = analyzer.stability_analysis(poles, zeros)
    
    # Neural network dynamics
    print("\n🧠 NEURAL NETWORK DYNAMICS...")
    t, response = analyzer.analyze_network_dynamics(num_neurons=10, t_final=2)
    print(f"Simulated {response.shape[1]} neurons over {len(t)} time points")
    
    # Generate report
    print("\n📝 GENERATING ANALYSIS REPORT...")
    
    report = f"""
    ======================================================================
        INVERSE LAPLACE TRANSFORM - AI SYSTEM DYNAMICS REPORT
    ======================================================================
    
    TRANSFER FUNCTIONS ANALYZED
    ---------------------------
    1. First Order (Neuron):        H(s) = 1/(τs + 1)
    2. Second Order (Learning):     H(s) = ω_n²/(s² + 2ζω_n s + ω_n²)
    3. PID Controller:              H(s) = Kp + Ki/s + Kd·s
    
    SYSTEM PROPERTIES
    ----------------
    • Neuron Time Constant: τ = 0.01 seconds (10ms)
    • Neural Network: {response.shape[1]} interconnected neurons
    • System Stability: {'STABLE' if stable else 'UNSTABLE'}
    
    RESPONSE CHARACTERISTICS
    ------------------------
    • Rise Time (10-90%): ~2.2τ = 22ms
    • Settling Time (2%): ~4τ = 40ms
    • Peak Overshoot: Varies with damping ratio
    
    APPLICATIONS IN AI
    -----------------
    ✓ Neural Activation Modelling
    ✓ Learning Rate Optimization
    ✓ Feedback Control Systems
    ✓ Recurrent Network Analysis
    ✓ Stability Verification
    
    KEY INSIGHTS
    ------------
    • Laplace domain analysis provides efficient stability assessment
    • Inverse transform reconstructs interpretable time-domain behavior
    • First-order models capture basic neural response characteristics
    • Second-order systems model oscillatory learning dynamics
    
    ======================================================================
    """
    
    print(report)
    
    # Save report
    with open('ai_dynamics_report.txt', 'w') as f:
        f.write(report)
    
    print("\n✅ Analysis complete!")
    print("📄 Report saved as 'ai_dynamics_report.txt'")
    print("📊 Plots saved in 'plots/' directory")


if __name__ == "__main__":
    main()
