import argparse
from itertools import islice, chain
import math
import random
import subprocess
import sys
import time
import numpy as np

def is_prime(n, k=10):
    if n < 2: return False
    primes = [2,3,5,7,11,13,17,19,23,29]
    if any(map(lambda p: n % p == 0, primes)):
        return n in primes

    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2

    for _ in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True

def factorise_backup(n, b1=10000):
    if -1 <= n <= 1: return [n]
    if n < -1: return [-1] + factorise_backup(-n)
    wheel = [1,2,2,4,2,4,2,4,6,2,6]
    w = 0
    f = 2
    fs = []
    while f * f <= n and f < b1:
        while n % f == 0:
            fs.append(f)
            n //= f
        f += wheel[w]
        w += 1
        if w == 11: w = 3
    if n == 1: return fs
    c = 1
    while not is_prime(n):
        h = 1
        t = 1
        g = 1
        while g == 1:
            h = (h ** 2 + c) % n
            h = (h ** 2 + c) % n
            t = (t ** 2 + c) % n
            g = math.gcd(t - h, n)
        if is_prime(g):
            while n % g == 0:
                fs.append(g)
                n //= g
        else:
            # TODO: Does this ever happen?
            gfs = factorise_backup(g)
            print(f" ### Found composite {g}, {gfs}, {n%g}")
            assert n % g == 0
            while n % g == 0:
                fs += gfs
                n //= g
        c += 1
    fs.append(n)
    return fs


def factorise(n):
    timeout = 180
    if timeout > 0:
        try:
            result = subprocess.check_output(["factor", str(n)], timeout=timeout, text=True)
            return list(map(int, result.split(' ')[1:]))
        except subprocess.TimeoutExpired:
            pass
        print(f"Oh no!  Commandline `factor` couldn`t handle {n}.", flush=True, file=sys.stderr)
    return factorise_backup(n)


def matpow(b, i, m):
    p = np.identity(b.shape[0], dtype=b.dtype)
    while i > 0:
        i, z = divmod(i, 2)
        if z != 0:
            p = p @ b % m
        b = b @ b % m
    return p


FACTORS = [2, 3, 5, 7]
def test_poly(poly, base):
    global FACTORS
    def mkmat(poly):
        return np.array( [
            [ int(i == j) for i in range(1, len(poly)) ] + [poly[-j-1]] for j in range(len(poly))
        ])
    def isidentity(mat):
        return np.all(np.equal(np.identity(len(poly)), mat))

    mat = mkmat(poly)
    period = base ** len(poly) - 1

    if not isidentity(matpow(mat, period, base)):
        return False

    factors = period
    while True:
        for factor in FACTORS:
            if factors % factor == 0:
                if isidentity(matpow(mat, period // factor, base)):
                    return False
                while factors % factor == 0: factors //= factor
        if factors == 1: break
        FACTORS = sorted(set(factorise(period)))
    return True


def bitperm(length, bits):
    mask = (1 << bits) - 1
    x = mask
    while x >> length == 0:
        yield x
        x += (x & -x)
        x |= mask >> x.bit_count()


def randrange(a, b = None, c = None):
    if b is None:
        count = a
        a = 0
        c = 1
    elif c is None:
        count = b - a
        c = 1
    else:
        count = b - a
    count //= c

    x = count * 3 // 1
    step = int(count * 0.61803398875)
    while math.gcd(step, count) > 1:
        step -= 1
    for _ in range(count):
        x = (x + step) % count
        yield a + x * c


def mkpoly(x, bitmap, base, length):
    mod = base - 1
    result = []
    for _ in range(length):
        if bitmap & 1 != 0:
            o = x % mod + 1
            x //= mod
        else:
            o = 0
        bitmap >>= 1
        result.insert(0, o)
    return result

def one_large(bits, base, length, limit=1000000):
    for bitmap in bitperm(length - 1, bits):
        zo = [ (bitmap >> i) & 1 for i in range(length - 1) ]
        zo.reverse()
        for i in islice(randrange(1, base), limit):
            yield zo + [i]
        if (bitmap & 1) != 0:
            for j in reversed(range(len(zo))):
                for i in islice(randrange(2, base), limit // length):
                    yield zo[:j] + [i] + zo[j:]


def many_large(bits, base, length, limit=1000000):
    for bitmap in bitperm(length - 1, bits):
        for x in islice(randrange((base - 1) ** (bits + 1)), limit):
            yield mkpoly(x, bitmap * 2 + 1, base, length)


def ratelimit(step):
    while True:
        tick = time.time() + step
        yield True
        while time.time() < tick:
            yield False


def search(base, length):
    global FACTORS
    slowprint = ratelimit(0.2)
    period = base ** length - 1
    print(f"Searching {base=} {length=}, {period=}", flush=True)
    FACTORS = sorted(set(factorise(period)))
    print(f"(factors={FACTORS})", flush=True)
    for gen in [ one_large, many_large ]:
        for bits in range(1, length):
            for poly in gen(bits, base, length):
                if next(slowprint):
                    print(f"trying: {poly}, {length}", end="  \x1b[K\r", file=sys.stderr)
                if test_poly(poly, base):
                    print(f"{base=}, {length=}, {poly=}")
                    break
            else:
                print(f"\nCould not find {gen.__name__}({bits=}, {base=}, {length=})", flush=True)


def main(args):
    for length in args.lengths:
        for p in map(int, args.bases):
            search(p, length)


def range_type(arg):
    def int_or_range(s):
        if '-' in s:
            start, end = map(int, s.split('-'))
            if start > end:
                raise ValueError("Invalid range: start must be less than or equal to end")
            print(f'{start=},{end=}')
            return range(start, end + 1)
        return [int(s)]
    return chain(*map(int_or_range, arg.split(',')))


parser = argparse.ArgumentParser(description="Find polynomials for non-binary LFSRs")
parser.add_argument('bases', nargs='+', type=int)
parser.add_argument('--length', dest='lengths', type=range_type, default=range(2,24))
main(parser.parse_args())
