# Get all prime numbers up to 100
primes = []
for possiblePrime in range(2, 100):
    isPrime = True
    for num in range(2, possiblePrime):
        if possiblePrime % num == 0:
            isPrime = False
            break
    if isPrime:
        primes.append(possiblePrime)

print('Prime numbers up to 100:', primes)