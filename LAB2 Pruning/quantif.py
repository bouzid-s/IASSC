import numpy as np

# Données
x_original = np.array([1000., 2.49, 6.53, 43.47, 13.02, 39.32, 4.43, 40.27, 
                       9.78, 44.05, 27.48, 30.76, 38.43, 2.49, 13.9, 12.09, 
                       5.72, 22.96, 15.15, 16.05, 26.33, 47.6, 18.03, 18.21, 
                       16.69, 41.31, 7.9, 38.47, 23., 10.45, 9.82, 7.56, 
                       21.3, 9.32, 49.76, 28.36, 41.21, 28.13, 16.02, 33.81, 
                       35.65, 25.12, 18.87, 31.41, 38.88, 2.74, 42.54, 43.21, 
                       11.81, 31.88])

n_bits = 8
levels = 2**n_bits

def quantize_percentile(x, n_bits, p_low, p_high):
    p_min = np.percentile(x, p_low)
    p_max = np.percentile(x, p_high)
    if p_min >= p_max:
        raise ValueError(f"Invalid percentiles: p_min={p_min}, p_max={p_max}")
    scale = (p_max - p_min) / (levels - 1)
    zero_point = -p_min / scale
    x_q = np.clip(np.round(x / scale + zero_point), 0, levels - 1).astype(np.int32)
    return x_q, scale, zero_point

def calculate_errors(x_original, x_dequant):
    mse = np.mean((x_original - x_dequant) ** 2)
    mae = np.mean(np.abs(x_original - x_dequant))
    return mse, mae

# Trouver les percentiles optimaux
best_p_low, best_p_high = 0, 0
min_mse = float('inf')
for p_low in range(0, 10):  # Tester les percentiles bas (0 à 10%)
    for p_high in range(90, 100):  # Tester les percentiles hauts (90 à 100%)
        try:
            x_q, scale, zero_point = quantize_percentile(x_original, n_bits, p_low, p_high)
            x_dequant = (x_q - zero_point) * scale
            mse, mae = calculate_errors(x_original, x_dequant)
            if mse < min_mse:
                min_mse = mse
                best_p_low, best_p_high = p_low, p_high
        except ValueError:
            continue

print(f"Meilleurs percentiles: p_low={best_p_low}%, p_high={best_p_high}%")
print(f"Erreur quadratique moyenne minimale (MSE): {min_mse:.4f}")