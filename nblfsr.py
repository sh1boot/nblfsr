import argparse
from itertools import islice, chain
import math
import random
import subprocess
import sys
import time
import numpy as np
import requests

def is_prime(n, k=30):
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

def factorise_backup(n, b1=10000, b2=1000000):
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
        h = 3
        t = 3
        g = 1
        while g == 1:
            h = (h ** 2 + c) % n
            h = (h ** 2 + c) % n
            t = (t ** 2 + c) % n
            g = math.gcd(t - h, n)
            b2 -= 1
            if b2 == 0: raise OverflowError(f"Too hard to factor {n}")
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
                fs.extend(gfs)
                n //= g
        c += 1
    fs.append(n)
    return fs


FACTORLIST = {}
def loadfactors():
    with open('factorlist.txt', 'rt') as f:
        for line in f:
            line = line.split('#', 1)[0]
            value, factors = line.split(':')
            factors = factors.strip().split(' ')
            value = int(value)
            factors = tuple(map(int, factors))
            product = math.prod(factors)
            if value != product:
                print(f"Incorrect factors of {value}: {factors} -- sed -e 's/^{value}: /{product}: /'", file=sys.stderr)
                print(f"{product}: {factors}")
                value = product
            FACTORLIST[value] = factors


def _factorise_db(n):
    endpoint = 'https://factordb.com/api'
    reply = requests.get(endpoint, params={"query": str(n)})
    if reply.status_code != 200:
        raise OverflowError(f"factordb query={n} returned {reply.status_code}: {reply=}")
    result = reply.json()
    status = result['status']
    if status != 'FF' and status != 'P' and status != 'PRP':
        raise OverflowError(f"factordb has not fully factored {n}: {status=}, {result['id']=}")
    factors = result['factors']
    print(f"factordb: {status=}, {n=}, {result['id']=}, {factors=}", end=' ', file=sys.stderr)
    factors = tuple(int(x) for x, p in factors for _ in range(p))
    print(f"{factors=}", file=sys.stderr)
    return factors


FACTOR_TIMEOUT = 180
def _factorise_1(n):
    if FACTOR_TIMEOUT < 0:
        raise OverflowError("skipped")
    if FACTOR_TIMEOUT > 0:
        try:
            factors = subprocess.check_output(["factor", str(n)], timeout=FACTOR_TIMEOUT, text=True)
            return tuple(map(int, factors.split(' ')[1:]))
        except subprocess.CalledProcessError as e:
            print(f"factor command failed: {e}")
        except subprocess.TimeoutExpired:
            print(f"timeout while waiting for factor {n}", flush=True, file=sys.stderr)
    try:
        return tuple(sorted(_factorise_db(n)))
    except OverflowError as e:
        print(f"factordb failed: {e}")
    try:
        return tuple(sorted(factorise_backup(n)))
    except OverflowError as e:
        # see if the list has been updated externally with something helpful
        loadfactors()
        if n not in FACTORLIST:
            raise e
        return FACTORLIST[n]


def _factorise_p(base, exp):
    def allfactors(n):
        halfway = int(math.sqrt(n))
        for i in range(2, halfway):
            if n % i == 0: yield i
        for i in range(halfway, 1, -1):
            if n % i == 0: yield n // i

    seen = set()
    factors = []
    n = base ** exp - 1
    for f in allfactors(exp):
        newprimes = set(factorise(base, f)) - seen
        seen |= newprimes
        for p in newprimes:
            while n % p == 0:
                n //= p
                factors.append(p)
        if n == 1: break
    if n > 1: factors.extend(factorise(n))
    factors.sort()
    return factors


def factorise(base, exp=None):
    if len(FACTORLIST) == 0:
        loadfactors()

    n = base ** exp - 1 if exp else base
    if n in FACTORLIST:
        return FACTORLIST[n]

    if exp is None:
        factors = _factorise_1(n)
    else:
        try:
            factors = _factorise_p(base, exp)
        except OverflowError:
            factors = _factorise_db(f"{base}^{exp}-1")

    assert math.prod(factors) == n
    FACTORLIST[n] = factors
    if exp or n > 0x10000000000000000:
        with open('newfactors.txt', 'at') as f:
            print(f'{n}: {" ".join(map(str, factors))}', file=f)
    return factors


def matpow_wide(b, i, m):
    m = np.uint64(m)
    m24 = np.uint64(1 << 24) % m
    m48 = np.uint64(m24 << np.uint64(24)) % m
    def matmul(a, b, m):
        alo = a & np.uint64(0x00ffffff)
        ahi = a >> np.uint64(24)
        blo = b & np.uint64(0x00ffffff)
        bhi = b >> np.uint64(24)
        lo = alo @ blo
        md = alo @ bhi + ahi @ blo
        hi = ahi @ bhi
        md += lo >> np.uint64(24)
        lo &= np.uint64(0x00ffffff)
        hi += md >> np.uint64(24)
        md &= np.uint64(0x00ffffff)
        md = md * m24 % m
        hi = hi * m48 % m
        return (lo + md + hi) % m

    p = np.identity(b.shape[0], dtype=b.dtype)
    while i > 0:
        i, z = divmod(i, 2)
        if z != 0:
            p = matmul(p, b, m)
        b = matmul(b, b, m)
    return p

SHUSH = 0
def matpow(b, i, m):
    if (m - 1) * (m - 1) * b.shape[0] > 0xffffffffffffffff:
        global SHUSH
        if SHUSH != m:
            print(f"Doing {b.shape[0]}^2 matrix mod {m} the hard way.", file=sys.stderr, flush=True)
            SHUSH = m
        return matpow_wide(b, i, m)

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
    def isidentity(mat):
        return np.all(np.equal(np.identity(len(poly)), mat))

    mat = np.array([
            [ int(i == j) for i in range(1, len(poly)) ] + [poly[-j-1]] for j in range(len(poly))
        ], dtype=np.uint64)
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
        # Not expected to be needed provided the global has been set up
        # beforehand.
        FACTORS = sorted(set(factorise(base, len(poly))))
    return True


def bitperm(length, bits):
    mask = (1 << bits) - 1
    x = mask
    while x >> length == 0:
        yield x
        x += (x & -x)
        x |= mask >> x.bit_count()


def randrange(a, b = None, c = None, *, seed=1):
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

    x = (count + 5 * seed) * 31 // 101
    step = (count - seed) * 0x9e3779b97f4a7bb5 // 0xffffffffffffffc5
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


def one_large(bits, base, length, inner_limit=1000):
    for bitmap in bitperm(length - 1, bits):
        zo = [ (bitmap >> i) & 1 for i in range(length - 1) ]
        zo.reverse()
        for i in islice(randrange(1, base, seed=bitmap), inner_limit):
            yield zo + [i]


def many_large(bits, base, length, inner_limit=1000):
    for bitmap in bitperm(length - 1, bits):
        for x in islice(randrange((base - 1) ** (bits + 1), seed=bitmap), inner_limit):
            yield mkpoly(x, bitmap * 2 + 1, base, length)


def ratelimit(step):
    while True:
        tick = time.time() + step
        yield True
        while time.time() < tick:
            yield False


def getfactors(base, length):
    period = base ** length - 1
    print(f"Factorising {base}^{length}-1: {period}", file=sys.stderr, flush=True)
    try:
        factors = factorise(base, length)
        print(f"{base}^{length}-1={period}: {' '.join(map(str, factors))}", flush=True)
    except OverflowError as e:
        print(f"{base}^{length}-1={period}: Overflow: {e}", flush=True)


def search(base, length):
    global FACTORS
    slowprint = ratelimit(0.2)
    period = base ** length - 1
    print(f"Searching {base=} {length=}, {period=}", flush=True)
    try:
        FACTORS = sorted(set(factorise(base, length)))
        print(f"(factors={FACTORS})", flush=True)
    except OverflowError as e:
        print(f"Overflow: {e}")
        return

    for gen in [ one_large, many_large ]:
        for bits in range(1, length):
            bailout = 10000
            for poly in islice(gen(bits, base, length), bailout):
                if next(slowprint):
                    print(f"trying: {poly}, {bits}", end="  \x1b[K\r", file=sys.stderr)
                if test_poly(poly, base):
                    print(f"{base=}, {length=}, {poly=}, {bits=}", flush=True)
                    break
            else:
                print(f"\nCould not find {gen.__name__}({bits=}, {base=}, {length=})", flush=True)


def main(args):
    global FACTOR_TIMEOUT
    FACTOR_TIMEOUT = args.factor_timeout
    for length in args.lengths:
        for p in map(int, args.bases):
            if not args.ceiling or p ** (length - 1) < args.ceiling:
                if args.factorise:
                    getfactors(p, length)
                else:
                    search(p, length)


def range_type(arg):
    def int_or_range(s):
        if '-' in s:
            start, end = map(int, s.split('-'))
            if start > end:
                raise ValueError("Invalid range: start must be less than or equal to end")
            return range(start, end + 1)
        return [int(s)]
    return chain(*map(int_or_range, arg.split(',')))


parser = argparse.ArgumentParser(description="Find polynomials for non-binary LFSRs")
parser.add_argument('bases', nargs='+', type=int)
parser.add_argument('--length', dest='lengths', type=range_type, default=range(2,24))
parser.add_argument('--factorise', default=False, action='store_true')
parser.add_argument('--factor-timeout', type=int, default=180)
parser.add_argument('--ceiling', type=eval, default=None)  # TODO: something less dangerous
main(parser.parse_args())
