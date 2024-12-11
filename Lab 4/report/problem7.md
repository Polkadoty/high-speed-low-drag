# Lab 4 Equations and Calculations

## 1. Oblique Shock Relations

### θ-β-M Relation
$$\tan \theta = 2 \cot \beta \frac{M_1^2 \sin^2 \beta - 1}{M_1^2(\gamma + \cos 2\beta) + 2}$$

### Property Ratios Across Shock
1. Pressure Ratio:
$$\frac{P_2}{P_1} = \frac{2\gamma M_{n1}^2 - (\gamma-1)}{\gamma+1}$$

2. Temperature Ratio:
$$\frac{T_2}{T_1} = \frac{[2\gamma M_{n1}^2 - (\gamma-1)][({\gamma-1})M_{n1}^2 + 2]}{(\gamma+1)^2M_{n1}^2}$$

3. Density Ratio:
$$\frac{\rho_2}{\rho_1} = \frac{(\gamma+1)M_{n1}^2}{(\gamma-1)M_{n1}^2 + 2}$$

4. Stagnation Pressure Ratio:
$$\frac{P_{02}}{P_{01}} = \left(\frac{(\gamma+1)M_{n1}^2}{(\gamma-1)M_{n1}^2 + 2}\right)^{\gamma/(\gamma-1)} \left(\frac{\gamma+1}{2\gamma M_{n1}^2 - (\gamma-1)}\right)^{1/(\gamma-1)}$$

### Velocity Behind Shock
$$V_2 = V_1\sqrt{\frac{(\gamma-1)M_{n1}^2 + 2}{2\gamma M_{n1}^2 - (\gamma-1)}}$$

## 2. Viscosity Calculation (Sutherland's Law)
$$\mu = \mu_0 \left(\frac{T}{T_0}\right)^{3/2} \frac{T_0 + S}{T + S}$$
where:
- μ₀ = 1.716×10⁻⁵ kg/m-s
- T₀ = 273.15 K
- S = 110.4 K

## 3. Reynolds Number
$$Re_L = \frac{\rho V L}{\mu}$$

## 4. Recovery Temperature
$$T_r = T_{\infty}\left(1 + r\frac{\gamma-1}{2}M_{\infty}^2\right)$$
where r = 0.896 for turbulent flow

## 5. Heat Transfer Calculations

### Nusselt Number (Turbulent Flow)
$$Nu_L = 0.0296 \cdot Re_L^{4/5} \cdot Pr^{1/3}$$

### Heat Transfer Coefficient
$$h = \frac{Nu_L \cdot k}{L}$$

### Heat Flux
$$q = h(T_r - T_s)$$

## 6. Uncertainty Analysis

For sequential perturbation:
$$\delta X = \sqrt{\sum_{i=1}^n \left(\frac{X(x_i + \delta x_i) - X(x_i - \delta x_i)}{2}\right)^2}$$

For root sum square:
$$\delta X = \sqrt{\sum_{i=1}^n \left(\frac{\partial X}{\partial x_i}\delta x_i\right)^2}$$

## Constants Used
- γ = 1.4 (ratio of specific heats for air)
- R = 287 J/kg-K (gas constant for air)
- Pr = 0.7 (Prandtl number for air)
- k = 0.02 W/m-K (thermal conductivity of air)