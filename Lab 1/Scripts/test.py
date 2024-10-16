import numpy as np
import matplotlib.pyplot as plt

# Define parameters
A = 1  # Amplitude
T = 2 * np.pi  # Period
t = np.linspace(0, 2 * T, 1000)  # Time array
zeta = 0.2  # Damping ratio
omega_n = np.pi / T  # Natural frequency

# Function to calculate the Fourier series approximation
def square_wave_approximation(t, A, T, harmonics):
    f_t = np.zeros_like(t)
    for n in range(1, harmonics + 1):  # Use harmonics up to the given number
        f_t += (4 * A / (np.pi * (2 * n - 1))) * np.cos((2 * n - 1) * 2 * np.pi * t / T)
    return f_t

# Function to calculate the response of the mass-spring-damper system
def system_response(t, A, T, harmonics, zeta, omega_n):
    response = np.zeros_like(t)
    for n in range(1, harmonics + 1):  # Use harmonics up to the given number
        omega = (2 * n - 1) * 2 * np.pi / T
        f_n = (4 * A / (np.pi * (2 * n - 1)))  # Amplitude of nth harmonic
        H = 1 / np.sqrt((1 - (omega / omega_n) ** 2) ** 2 + (2 * zeta * omega / omega_n) ** 2)  # Magnitude of transfer function
        phase = np.arctan2(2 * zeta * omega / omega_n, 1 - (omega / omega_n) ** 2)  # Phase angle
        response += H * f_n * np.cos(omega * t - phase)
    return response

# Plotting the approximations
harmonics_list = [3, 5, 10]
plt.figure(figsize=(15, 15))

for i, harmonics in enumerate(harmonics_list):
    plt.subplot(3, 2, 2 * i + 1)
    f_t = square_wave_approximation(t, A, T, harmonics)
    plt.plot(t, f_t, label=f'First {harmonics} Harmonics (Square Wave Approximation)')
    plt.title(f'Approximation of Square Wave using First {harmonics} Harmonics')
    plt.xlabel('Time t')
    plt.ylabel('f(t)')
    plt.grid(True)
    plt.legend()

    plt.subplot(3, 2, 2 * i + 2)
    response = system_response(t, A, T, harmonics, zeta, omega_n)
    plt.plot(t, response, label=f'System Response with First {harmonics} Harmonics')
    plt.title(f'System Response using First {harmonics} Harmonics')
    plt.xlabel('Time t')
    plt.ylabel('Response')
    plt.grid(True)
    plt.legend()

plt.tight_layout()
plt.show()