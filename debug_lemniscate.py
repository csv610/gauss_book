from python.lemniscate import sl, cl, K, GAUSS_CONSTANT, lemniscate_integral, lemniscate_integral_inverse
import math

print('K =', K)
print('π*G/2 =', math.pi*0.8346268416740732/2)
print('K =', K)

print('\nsl(0) =', sl(0))
print('sl(K/2) =', sl(K/2))
print('sl(K) =', sl(K))
print('sl(2K) =', sl(2*K))

print('\ncl(0) =', cl(0))
print('cl(K/2) =', cl(K/2))
print('cl(K) =', cl(K))

# Test integral
u = 0.5
omega = lemniscate_integral(u)
print(f'\nlemniscate_integral({0.5}) = {omega}')
u_rec = lemniscate_integral_inverse(omega)
print(f'lemniscate_integral_inverse({omega}) = {u_rec}')

# Check sl(1)
print('\nsl(1) =', sl(1))
print('sl(2) =', sl(2))
print('sl(3) =', sl(3))

# Check sl(K/2) explicitly
K_half = K/2
print(f'\nK/2 = {K/2}')
print(f'sl(K/2) = {sl(K/2)}')
print(f'1/sqrt(2) = {1/math.sqrt(2)}')

# Test the integral at u = 1/sqrt(2)
u_test = 1/math.sqrt(2)
omega = lemniscate_integral(u_test)
print(f'\nlemniscate_integral(1/sqrt(2)) = {omega}')
print(f'K/2 = {K/2}')

# Test sl at K/2
print(f'sl(K/2) = {sl(K/2)}')