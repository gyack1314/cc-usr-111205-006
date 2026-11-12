"""log域数值基础。

推断全程在 log 域进行：logsumexp 用 math.fsum 求和，结果与求值顺序无关；
正态 log CDF 在远左尾切换渐近展开，避免 erfc 下溢把正概率吞成零；
区间概率用 log 域差（expm1），不在线性空间相减，极小区间概率不会因
相减下溢被误判为不可能。
"""
import math

LOG_2PI = math.log(2.0 * math.pi)
_INV_SQRT2 = 1.0 / math.sqrt(2.0)
# erfc(|x|/sqrt(2)) 在 |x| 约 38.6 处才下溢为 0，留余量取 38 切换渐近展开。
_ERFC_LIMIT = 38.0


def logsumexp(values):
    """log(sum(exp(v)))，空集或全 -inf 返回 -inf。"""
    vals = list(values)
    if not vals:
        return -math.inf
    m = max(vals)
    if m == -math.inf:
        return -math.inf
    return m + math.log(math.fsum(math.exp(v - m) for v in vals))


def logadd(a, b):
    """log(exp(a) + exp(b))。"""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a < b:
        a, b = b, a
    return a + math.log1p(math.exp(b - a))


def log_norm_cdf(x):
    """log Phi(x)，全实数轴有效。

    x >= 0 用 log1p(-0.5*erfc(x/sqrt2)) 保住接近 1 的精度；
    -38 <= x < 0 用 log(0.5*erfc(-x/sqrt2))；
    x < -38 时 erfc 会下溢，改用渐近展开
        Phi(x) ≈ phi(x)/(-x) * sum_k (-1)^k (2k-1)!! x^(-2k)
    该级数在 |x| >= 38 处截断误差远小于双精度分辨率。
    """
    if math.isnan(x):
        raise ValueError('需要有限数值')
    if x == math.inf:
        return 0.0
    if x == -math.inf:
        return -math.inf
    if x >= 0.0:
        return math.log1p(-0.5 * math.erfc(x * _INV_SQRT2))
    if x >= -_ERFC_LIMIT:
        return math.log(0.5 * math.erfc(-x * _INV_SQRT2))
    log_phi = -0.5 * x * x - 0.5 * LOG_2PI
    inv_x2 = 1.0 / (x * x)
    series = 1.0
    term = 1.0
    for k in range(1, 1000):
        term *= -(2 * k - 1) * inv_x2
        series += term
        if abs(term) < 1e-18:
            break
    return log_phi - math.log(-x) + math.log(series)


def log_gaussian_pdf(y, mu, sd):
    """正态密度的 log（exact 观测的发射项）。"""
    z = (y - mu) / sd
    return -0.5 * z * z - math.log(sd) - 0.5 * LOG_2PI


def log_interval_prob(left, right, mu, sd):
    """log P(left < Y <= right)，log 域差，不做线性相减。

    两端 log CDF 在浮点下不可分辨（区间窄于尾部分辨率）时返回 -inf，
    这是诚实的零支持，而不是下溢造成的假不可能。
    """
    lu = log_norm_cdf((right - mu) / sd)
    ll = log_norm_cdf((left - mu) / sd)
    if lu == -math.inf:
        return -math.inf
    d = ll - lu
    if d == 0.0:
        return -math.inf
    return lu + math.log(-math.expm1(d))


def log_upper_tail(left, mu, sd):
    """log P(Y > left)，量程上限右删失观测的发射项。"""
    return log_norm_cdf((mu - left) / sd)
