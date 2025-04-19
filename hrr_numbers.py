import numpy as np


class HRRNumbers:
    def __init__(self, N:int, beta:float=25):
        '''Implements integer-like unitary Holographic Reduced Representations
        that respect modular arithmetic operations.
        
        Parameters
        ----------
        N: int
            The least upper bound of the absolute value of representable numbers.
            Due to the construction process, the resulting object is likely to be able 
            to represent more than this bound suggests.

        beta: float
            Controls the entropy of the output of the softmax retrieval function used
            in the mapping process. Generally this number should be large when the amount 
            of representable numbers is large to ensure operations produce reliable outputs.
        '''
        p = HRRNumbers._primes()
        self.primes = []
        while not self.primes or np.prod(self.primes) < N:
            self.primes.append(next(p))

        self.beta = beta
        self.mtable, self.atable = self._build_maps()

    def encode(self, n:int):
        # Check if value can be faithfully represented
        assert abs(n) < np.prod(self.primes)
        # Encodes number as its set of residues (mod p_1, p_2, ...)
        hrr = np.exp(2j*np.pi*np.array([(n%p)/p for p in self.primes]))
        return HRRInstance(self, hrr)
    
    def decode(self, hrr, residues=False):
        # Extended Euclidean algorithm/Bezout's identity/Chinese remainder theorem
        def _solve(a, b):
            if a == 0:
                return 0, 1
            x0, y0 = _solve(b % a, a)
            x = y0 - (b // a) * x0
            y = x0
            return x, y
        theta = np.angle(hrr.value)
        theta[theta<0]+=2*np.pi  # remap from [-pi/pi) to [0,2pi)
        rem = (self.primes*theta/(2*np.pi)).round().astype(int)
        rem = rem % self.primes
        if len(self.primes) == 1:
            return rem[0]
        if residues:
            return rem
        
        prod = np.prod(self.primes)
        res = 0
        for i in range(len(self.primes)):
            prod_i = prod // self.primes[i]
            coeff, _ = _solve(prod_i, self.primes[i])
            res += rem[i] * prod_i * coeff

        return res % prod
    
    @property
    def max_value(self): 
        return np.prod(self.primes)
    
    def map_to_space(self, x, origin=None):
        m0 = self.mtable
        m1 = self.atable
        if origin == 'mul':
            m0, m1 = m1, m0
        exp = np.exp(self.beta*np.einsum('ij,j->ij', m0, x.conj()).real)
        weights = np.einsum('ij,j->ij', exp, 1/np.sum(exp, axis=0))
        proj = np.einsum('ij,ij->j', weights, m1)
        return proj

    def __call__(self, n:int):
        return self.encode(n)

    def _build_maps(self):
        # Find a generator g of the multiplicative group of Fp for each prime p
        # Construct an ordered list of p-1 generator powers -> g^1, g^2, ..., g^{p-1}
        logarithms = []
        for p in self.primes:
            if p == 2:
                logarithms.append([1])
                continue
            for n in range(2, p):
                glist = []
                gset = set()
                for k in range(1, p):
                    m = n**k % p
                    if m in gset:
                        break
                    glist.append(m)
                    gset.add(m)
                if len(glist) == p-1:
                    logarithms.append(glist)
                    break
        # Construct isomorphism between multiplicative group of F_p and additive group Z/(p-1)Z
        # Everything can be represented on the complex unit circle + {0} i.e., via HRRs
        # Additive identity of F_p doesn't map to anything in Z/(p-1)Z, so gets sent to 0
        mtable = np.zeros((np.max(self.primes)+1, len(self.primes)), dtype=np.complex128)
        mtable[0,:] = np.ones(mtable.shape[-1])
        for i, p in enumerate(self.primes):
            logs = np.array(logarithms[i])
            mtable[1:p,i] = np.exp(2j*np.pi*logs/p)
        mtable[-1,:] = np.ones(len(self.primes))  # sufficient to map 0 back to identity 1

        atable = np.zeros((np.max(self.primes)+1, len(self.primes)), dtype=np.complex128)
        for i, p in enumerate(self.primes):
            atable[1:p,i] = [np.exp(2j*np.pi*k/(p-1)) for k in range(1,p)]
        
        return mtable, atable
    
    def _primes():
        # Basic prime generator
        yield 2
        primes = [2]
        i = 3
        while True:
            for p in primes:
                if p**2 > i:
                    primes.append(i)
                    yield i
                    break
                if i%p == 0:
                    break
            i += 1


class HRRInstance:
    def __init__(self, parent:HRRNumbers, value:np.ndarray):
        '''An object representing a number.
        Class implements arithmetic operations between objects.
        '''
        self.parent = parent
        self.value = value
    
    def decode(self, residues=False):
        return self.parent.decode(self, residues=residues)
    
    def __add__(self, other):
        if self.parent == other.parent:
            res = self.value * other.value
            return HRRInstance(self.parent, res)
        else:
            raise ValueError('Operands must share same parent object.')

    def __mul__(self, other, divide=False):
        if self.parent == other.parent:
            x = self.parent.map_to_space(self.value)
            y = self.parent.map_to_space(other.value)
            if divide: 
                y = y.conj()
            res = self.parent.map_to_space(x*y, origin='mul')
            return HRRInstance(self.parent, res)
        else:
            raise ValueError('Operands must share same parent object.')
        
    def __neg__(self):
        return HRRInstance(self.parent, self.value.conj())
    
    def __sub__(self, other):
        return self + (-other)
    
    def __truediv__(self, other):
        return self.__mul__(other, divide=True)

