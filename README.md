Integer-like HRR representations that respect modular arithmetic operations: $+$, $-$, $\times$.

This repo was largely motivated as a proof of concept demonstrating a particular model of modular arithmetic geared towards computational cognitive science applications. As designed, all representations and their operations can be implemented as neural networks and are based on well-established models of cognition, mental representation, and memory, including [Holographic Reduced Representations](https://ieeexplore.ieee.org/document/377968) and [modern Hopfield networks](https://arxiv.org/abs/2008.02217).

### Basic Usage
```
from hrr_numbers import HRRNumbers

# Initialize class such that integers of at least 25 are faithfully represented.
# Beta parameter influences retrieval (higher = less noise)
numbers = HRRNumbers(25, beta=10)

print(numbers.max_value)
# Output: 30

# See encoded finite fields
print(numbers.primes)
# Output: [2, 3, 5]

# Instantiate a particular element
a = numbers(7)
print(a.decode())
# Output: 7

# Get list of residues (mod primes as above)
print(a.decode(residues=True))
# Output: [1 1 2]

b = numbers(3)
print((a+b).decode())
# Output: 10
print((a*b).decode())
# Output: 21
print((a-b).decode())
# Output: 4
```

### Background Details
Modular arithmetic has already been [implemented](https://pmc.ncbi.nlm.nih.gov/articles/PMC10659444/) using similar techniques and representational frameworks that may provide a computational account of grid cell network dynamics, which are thought to encode properties of physical space in an [abstract modular structure](https://pmc.ncbi.nlm.nih.gov/articles/PMC4042558/) for navigation purposes. The approach in this repo provides a tweak to how multiplication is implemented that is efficient in both time and computational resources as it doesn't require much of the machinery proposed in the literature. 

The central mathematical objects in this context governing modular arithmetic are the [cyclic groups](https://en.wikipedia.org/wiki/Cyclic_group), here denoted by $\mathbb{Z}/n\mathbb{Z}$, whose elements correspond to the subset of integers ($\mathbb{Z}$) from $0$ to $n-1$ and whose group operation can be realized as addition mod $n$ for some non-negative integer $n$. These subsets also inherit a multiplication operation from $\mathbb{Z}$&mdash;again, mod $n$&mdash;although they do not in general have unique inverses under multiplication so fail to satisfy the [group axioms](https://en.wikipedia.org/wiki/Group_(mathematics)). However when $p$ is prime, the set $\{1,2, \dots, p-1\}$ does permit a group structure under multiplication, so the resulting object formed from this set (including 0) under addition and multiplication is called a [finite field](https://en.wikipedia.org/wiki/Finite_field) $\mathbb{F}_p$ where all the usual arithmetic operations are well-defined.

Prime numbers are also relevant due to the prime factorization theorem, namely that every integer greater than 1 can be uniquely represented as a product of powers of prime numbers (e.g., $60=2^2\cdot 3\cdot 5$). Leveraging these facts, representational schemes have been constructed based on cyclic groups of order $p$ that share many (but not all) properties of the integers. Common to most models is a simple implementation of addition (mod $n$) based on multiplication between roots of unity on the complex unit circle.

The key insight in this work follows the well-established existence of an isomorphism between the multiplicative group $\mathbb{F}_p^\times$ of the finite field $\mathbb{F}_p$ and the cyclic group $\mathbb{Z}/(p-1)\mathbb{Z}$ for every $p$. Concretely, this means that there is a map $\psi:\mathbb{F}_p^\times\to\mathbb{Z}/(p-1)\mathbb{Z}$ such that $\psi(1)=0$ and for all $n,m\in\mathbb{F}_p^\times$, $\psi(n\cdot m)=\psi(n)+\psi(m)$. Further, there is an inverse map $\psi^{-1}$ satisfying $\psi^{-1}(0)=1$ and $\psi^{-1}(g+h)=\psi^{-1}(g)\cdot\psi^{-1}(h)$ for all $g,h\in\mathbb{Z}/(p-1)\mathbb{Z}$. 

In short, these maps are implemented in this work by a particular choice of weights in a modern Hopfield memory network such that multiplication can be computed using the exact same operations on the complex unit circle as addition (beyond the application of the maps themselves). The resulting structures are low in dimension yet capable of representing huge numbers, at the cost of introducing noise, which can be managed (for example) by increasing the retrieval parameter $\beta$.

### Notes
Although subtraction between representations is possible, because the structure is technically a direct product of cyclic groups, "negative" values wrap back from 0 to the maximum representable value. For example, if 6 is the maximum value, then -1 should decode to 5.

"Division" is also implemented *however* this does not correspond whatsoever to division between integers, which is not an operation that is even defined between integers in general (as the integers do not form a field). Instead, for all elements $x$ and $y$, division satisfies $(x/y)\times y=y\times(x/y)=x$. In other words, this operation computes a (one-sided) pseudoinverse. The only elements with a true inverse in this ring are the elements corresponding to the prime numbers, specifically those that are greater than the basis elements.
