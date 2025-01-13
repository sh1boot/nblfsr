import argparse
import builtins
from itertools import batched, chain, islice, repeat
import math
from operator import mul, eq

allzeroes = lambda v: not any(map(bool, v))

def _factors(n):
    """Factors an integer n into its prime factors."""
    factors = {}
    i = 2
    while i * i <= n:
        ct = 0
        while n % i == 0:
          ct += 1
          n //= i
        if ct > 0: factors[i] = ct
        i += 1
    if n > 1: factors[n] = 1
    return factors

def _period(base, length, *, n=1, punctured=False):
    return base ** (length * n) - int(punctured)

def _full_cycle(base, length, *, n=1, punctured=False, stop=False, gen=None):
  if gen is None: gen = repeat(None)
  if stop:
    period = _period(base, length, n=n, punctured=punctured) * int(stop)
    yield from islice(gen, period)
  else:
    yield from gen

def _fixedlen(v, length):
  return islice(chain(v, repeat(0)), length)

POLYS = {
# {{{
  (3, 2): [1, 1],
  (3, 3): [0, 1, 2],
  (3, 4): [0, 0, 1, 1],
  (3, 5): [0, 0, 0, 1, 2],
  (3, 6): [0, 0, 0, 0, 1, 1],
  (3, 7): [0, 0, 0, 0, 1, 0, 2],
  (3, 8): [0, 0, 0, 0, 1, 0, 0, 1],
  (3, 9): [0, 0, 0, 0, 1, 0, 0, 0, 2],
  (5, 2): [1, 3],
  (5, 3): [0, 1, 2],
  (5, 4): [0, 1, 1, 2],
  (5, 5): [0, 0, 0, 1, 2],
  (5, 6): [0, 0, 0, 0, 1, 3],
  (5, 7): [0, 0, 0, 0, 2, 4, 3],
  (5, 8): [0, 0, 0, 0, 0, 1, 1, 2],
  (5, 9): [0, 0, 0, 1, 0, 0, 0, 0, 2],
  (7, 2): [1, 4],
  (7, 3): [1, 0, 3],
  (7, 4): [1, 1, 0, 2],
  (7, 5): [0, 0, 0, 1, 3],
  (7, 6): [0, 0, 1, 0, 1, 2],
  (7, 7): [0, 0, 0, 0, 0, 1, 3],
  (7, 8): [0, 0, 0, 0, 0, 0, 1, 4],
  (11, 2): [1, 3],
  (11, 3): [1, 0, 2],
  (11, 4): [1, 0, 0, 3],
  (11, 5): [0, 0, 1, 0, 2],
  (11, 6): [0, 0, 1, 1, 0, 4],
  (11, 7): [0, 0, 0, 0, 1, 0, 2],
  (11, 8): [0, 0, 0, 1, 1, 0, 0, 3],
  (13, 2): [1, 11],
  (13, 3): [1, 0, 2],
  (13, 3): [1, 1, 7],
  (13, 4): [1, 1, 0, 2],
  (13, 4): [1, 1, 3, 2],
  (13, 5): [1, 0, 1, 0, 2],
  (13, 5): [1, 1, 1, 0, 2],
  (13, 5): [1, 1, 1, 2, 6],
  (13, 6): [0, 0, 1, 1, 0, 2],
  (13, 6): [1, 1, 0, 1, 0, 2],
  (13, 6): [1, 0, 1, 1, 1, 2],
  (13, 6): [1, 1, 1, 1, 1, 6],
  (13, 7): [0, 0, 1, 0, 0, 0, 2],
  (13, 7): [0, 1, 0, 1, 0, 0, 2],
  (13, 7): [0, 0, 0, 1, 1, 1, 2],
  (13, 7): [0, 1, 1, 1, 1, 0, 2],
  (13, 7): [0, 1, 1, 1, 1, 1, 2],
  (13, 7): [1, 1, 1, 1, 1, 1, 6],
  (13, 8): [0, 0, 0, 1, 0, 0, 1, 2],
# }}}
}

def _prime_1(base, length, punctured=True, stop=False, as_=next):
  """ The most basic LFSR-based generator. """
  poly = POLYS[(base, length)]
  shift = list(_fixedlen([1], length))

  # The expected period of an LFSR is base**length-1, but that -1 produces
  # a "punctured" de Bruijn sequence, which would cause problems in larger
  # sequences.  This function tweaks the output to make it the full period.
  if punctured == True:
    unpuncture = lambda s : s
  else:
    def unpuncture(s, k=1):
      # toggle between k,0,0,0.. and 0,0,0,..
      if allzeroes(s[1:]) and s[0] in (0, k):
        s = list(_fixedlen([k - s[0]], length))
      return s

  for _ in _full_cycle(base, length, stop=stop, punctured=punctured):
    yield as_(iter(shift))
    x = sum(map(mul, shift, poly)) % base
    shift = [x] + shift[:-1]
    shift = unpuncture(shift)


def _prime_n(base, n, length, punctured=False, stop=False, as_=next):
  assert n > 1  # Should call _prime_1() directly.
  # Map n values in the range 0..(base-1) to one value in 0..(base**n-1).
  def fuse(v):
    r = 0
    for d in v: r = r * base + d
    return r

  # step must be co-prime to period, but the period is a power of a single
  # prime, so a single bump always works.
  step = n if math.gcd(base, n) == 1 else (n + 1)

  # Because we're taking n steps per iteration, we must repeat the sequence n
  # times before quitting (unless we're never quitting).
  if stop: stop = int(stop) * step

  # We need to consume n consecutive outputs in the source generator so that we
  # can fuse them together to geth one value in the desired range.
  every_nth = lambda g: islice(g, None, None, step)

  # Punctured must be false here because otherwise we have no control over
  # the divisibility of the period and all our assumptions fall apart.
  gen = _prime_1(base, length * n, punctured=False, stop=stop, as_=iter)

  # And off we go!
  for shift in every_nth(gen):
    if punctured:
      shift = list(shift)
      if allzeroes(shift): continue
    shift = map(fuse, batched(shift, n))
    yield as_(shift)


def _generic(base, power, length, **kwargs):
  """ Delegate to appropriate generator function depending on parameters. """
  if base == 2:
    if power == 1:
      yield from _binary_1(length, **kwargs)
    else:
      yield from _binary_n(length, power,  **kwargs)
  else:
    if power == 1:
      yield from _prime_1(base, length, **kwargs)
    else:
      yield from _prime_n(base, power, length, **kwargs)

def generate(base, length, punctured=False, stop=False, as_=next):
  factors = _factors(base)
  n = len(factors)
  if n == 1:
    [(p, n)] = factors.items()
    yield from _generic(p, n, length, punctured=punctured, stop=stop, as_=as_)
    return
  fp = [ p ** i for (p, i) in factors.items() ]
  assert base == math.prod(fp)
  assert punctured == False  # too hard to think about

  def fuse(v):
    r = 0
    for f, d  in zip(fp, v): r = r * f + d
    return r

  def delegate(p, n):
    return _generic(p, n, length, punctured=punctured, stop=False, as_=iter)
  gens = [ delegate(p, n) for (p, n) in factors.items() ]

  for shift in _full_cycle(base, length, punctured=punctured, stop=stop, gen=zip(*gens)):
    yield as_(map(fuse, zip(*shift)))


def main(args):
  for base in args.base:
    for i in generate(base, args.length, stop=True, punctured=args.punctured, as_=args.as_):
      print(i)

def fn_type(arg):
  if arg == 'none': return lambda x: x
  try:
    return getattr(builtins, arg)
  except AttributeError as e:
    raise argparse.ArgumentTypeError(e)

parser = argparse.ArgumentParser(description="Example nblfsr generator code")
parser.add_argument('base', nargs='+', type=int, default=3)
parser.add_argument('--length', type=int, default=3)
parser.add_argument('--as', dest='as_', type=fn_type, default=next)
parser.add_argument('--punctured', action='store_true')
main(parser.parse_args())
