"""
Pytest configuration: mock external dependencies before any mk module is imported.

External deps (MP, yf_for, vpscyld, numba, yf_yld2000) are research/custom
packages not available via pip, so we mock them for CI testing.
"""
import sys
import types
import numpy as np
import pytest


def _install_mocks():
    """Install all external dependency mocks into sys.modules."""
    # -- MP --
    mp = types.ModuleType('MP')
    mp_mat = types.ModuleType('MP.mat')
    mp_mat_voigt = types.ModuleType('MP.mat.voigt')
    mp_mat_voigt.ijv = np.array([[0, 1, 2, 0, 2, 0],
                                  [0, 1, 2, 1, 1, 2]])
    mp_mat_voigt.vij = None
    mp_lib = types.ModuleType('MP.lib')
    mp_mpl = types.ModuleType('MP.lib.mpl_lib')
    mp_mpl.ticks_bins = lambda *a, **k: None
    mp_ssort = types.ModuleType('MP.ssort')
    mp_sh = types.ModuleType('MP.ssort.sh')
    mp_pb = types.ModuleType('MP.progress_bar')
    mp_pb.update_elapsed_time = lambda *a, **k: None
    mp_etc = types.ModuleType('MP.lib.etc')
    mp_etc.gen_hash_code2 = lambda nchar=6: 'abc123'
    mp_whichcomp = types.ModuleType('MP.lib.whichcomp')
    mp_whichcomp.determineEnvironment = lambda: ('', False)
    mp_temp = types.ModuleType('MP.lib.temp')
    mp_temp.gen_tempfile = lambda: '/tmp/test'
    mp_axes = types.ModuleType('MP.lib.axes_label')
    mp_axes.draw_guide = lambda *a, **k: None

    for name, mod in [
        ('MP', mp), ('MP.mat', mp_mat), ('MP.mat.voigt', mp_mat_voigt),
        ('MP.lib', mp_lib), ('MP.lib.mpl_lib', mp_mpl),
        ('MP.ssort', mp_ssort), ('MP.ssort.sh', mp_sh),
        ('MP.progress_bar', mp_pb), ('MP.lib.etc', mp_etc),
        ('MP.lib.whichcomp', mp_whichcomp), ('MP.lib.temp', mp_temp),
        ('MP.lib.axes_label', mp_axes),
    ]:
        sys.modules[name] = mod
    mp.mat = mp_mat
    mp.lib = mp_lib
    mp.ssort = mp_ssort
    mp.progress_bar = mp_pb
    mp_mat.voigt = mp_mat_voigt
    mp_lib.mpl_lib = mp_mpl
    mp_lib.etc = mp_etc
    mp_lib.whichcomp = mp_whichcomp
    mp_lib.temp = mp_temp
    mp_lib.axes_label = mp_axes

    # -- yf_for --
    yf_for = types.ModuleType('yf_for')

    def mock_vm(s):
        s = np.array(s, dtype=float)
        h = 0.5 * ((s[0] - s[1])**2 + s[0]**2 + s[1]**2 + 6 * s[5]**2)
        phi = h**0.5 if h > 0 else 0.
        dphi = np.zeros(6)
        d2phi = np.zeros((6, 6))
        if phi > 0:
            dphi[0] = (2 * s[0] - s[1]) / (2 * phi)
            dphi[1] = (2 * s[1] - s[0]) / (2 * phi)
        snew = s / phi if phi > 0 else s
        return snew, phi, dphi, d2phi

    yf_for.vm = mock_vm
    yf_for.gauss = lambda ndim, a, b: (b, a, b)
    yf_for.norme = lambda ndim, r: np.sqrt((r**2).sum())
    yf_for.hill48 = lambda s, f, g, h, n: (
        np.array(s, dtype=float), 1.0, np.zeros(6), np.zeros((6, 6)))
    yf_for.hqe = lambda s, r0, r90: (s, 1.0, np.zeros(3), np.zeros((3, 3)))
    yf_for.swift = lambda e, ks, n, e0, m, qq: (
        m, ks * (e + e0)**n, 0., 0., qq, n)
    yf_for.voce = lambda e, a, b0, c, b1, m, qq: (
        m, a - b0 * np.exp(-c * e) + b1 * e, 0., 0., qq)
    sys.modules['yf_for'] = yf_for

    # -- numba --
    numba = types.ModuleType('numba')
    numba.jit = lambda f=None, **k: (lambda x: x) if f is None else f
    sys.modules['numba'] = numba

    # -- vpscyld --
    vpscyld = types.ModuleType('vpscyld')
    vpscyld_lib = types.ModuleType('vpscyld.lib_dat')
    vpscyld_lib.xy2rt = lambda x, y: (x, y)
    vpscyld_lib.rad_xy2 = lambda *a: [1.]
    vpscyld_lib.rad_xy = lambda *a: 1.
    vpscyld_lib.nys_th = lambda *a, **k: (a[0], a[1])
    vpscyld_lib.pi_rad = lambda *a, **k: None
    vpscyld_cn = types.ModuleType('vpscyld.calc_normal')
    vpscyld_cn.main_var = lambda *a, **k: [0.5]
    vpscyld_cn.find_stress = lambda *a: ([0.], [0.])
    sys.modules['vpscyld'] = vpscyld
    sys.modules['vpscyld.lib_dat'] = vpscyld_lib
    sys.modules['vpscyld.calc_normal'] = vpscyld_cn
    vpscyld.lib_dat = vpscyld_lib
    vpscyld.calc_normal = vpscyld_cn

    # -- yf_yld2000 --
    yld2000 = types.ModuleType('yf_yld2000')
    yld2000.skew = lambda m, k, l, s: (
        np.array(s, dtype=float), 1., 1., np.zeros(6), 1., np.zeros((6, 6)))
    yld2000.readla = lambda fn: np.zeros(8)
    sys.modules['yf_yld2000'] = yld2000

    return mock_vm


# Install mocks at import time (before any test collection)
mock_vm = _install_mocks()


@pytest.fixture
def vm_func():
    """Provide the mock von Mises yield function."""
    return mock_vm
