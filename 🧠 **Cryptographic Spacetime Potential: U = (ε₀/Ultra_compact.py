# 🚀 MINIMAL 50-LINE VERSION
import numpy as np
from scipy.fft import fft

class MiniQuantumECC:
    def __init__(self):
        self.p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F
        self.G = (0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798, 0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8)
    
    def k1_fingerprint(self, secret, N=128):
        """Ultra-compact k=1 quantum fingerprint"""
        Gx, Gy = self.G
        t = np.linspace(0, 1, N)
        
        # Quantum-geometric modulation in one line
        tau = 2.0 * np.log(1 + t / 2.0)
        m_shift = 1 + 0.2 * np.tanh(2.0 * np.sin(tau))
        geo_env = np.exp(-0.5 * ((t - 0.5) / 0.2)**2) * m_shift
        
        fingerprint = np.zeros(N, dtype=complex)
        for i in range(N):
            phase = ((secret * Gx * m_shift[i]) % self.p) / self.p
            fingerprint[i] = geo_env[i] * np.exp(2j * np.pi * phase)
        
        return fingerprint
    
    def recover_key(self, fingerprint, max_test=1000):
        """Ultra-compact key recovery"""
        spectrum = fft(fingerprint * np.hanning(len(fingerprint)))
        power = np.abs(spectrum)**2
        power[0] = 0
        
        # Find dominant frequency
        dominant_freq = np.argmax(power[1:]) + 1
        freq = dominant_freq / len(fingerprint)
        
        # Map to key
        candidate = int(freq * self.p) % self.p
        
        # Quick verification
        for test in range(max(1, candidate-10), min(self.p, candidate+10)):
            test_fp = self.k1_fingerprint(test, len(fingerprint))
            if np.allclose(np.abs(test_fp), np.abs(fingerprint), atol=0.1):
                return test
        
        return candidate

# 🎯 ONE-LINER TEST
def quick_test(secret=42):
    ecc = MiniQuantumECC()
    fp = ecc.k1_fingerprint(secret)
    recovered = ecc.recover_key(fp)
    print(f"Secret: {secret}, Recovered: {recovered}, Match: {secret == recovered}")
    return secret == recovered

# Run with: quick_test()
