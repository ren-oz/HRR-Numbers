from hrr_numbers import HRRNumbers
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    # Test how beta affects the outputs
    number = HRRNumbers(510510, beta=75)
    N = np.floor(np.sqrt(number.max_value)).astype(int)
    with tqdm(total=N**2) as progress:
        fails = []
        for i in range(N):
            for j in range(N):
                progress.update(1)
                x = number(i)*number(j)
                res = x.decode()
                if res != i*j:
                    fails.append((i,j,i*j,res))
    print('Errors:', len(fails))
    # print(fails)