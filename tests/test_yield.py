"""Tests for mk.yieldFunction modules"""
import numpy as np
import pytest


class TestCpbLib:
    def test_deviator_ndarray(self):
        from mk.yieldFunction import cpb_lib
        result = cpb_lib.deviator(np.array([100., 200., 300., 0, 0, 0]))
        assert isinstance(result, np.ndarray)

    def test_deviator_list(self):
        from mk.yieldFunction import cpb_lib
        result = cpb_lib.deviator([1., 2., 3., 0, 0, 0])
        assert isinstance(result, np.ndarray)


class TestCpbIso:
    def test_main_ndarray(self):
        from mk.yieldFunction import cpb_iso
        result = cpb_iso.main(np.array([1., 0, 0, 0, 0, 0]))
        assert result is not None

    def test_main_list(self):
        from mk.yieldFunction import cpb_iso
        result = cpb_iso.main([1., 0, 0, 0, 0, 0])
        assert result is not None


class TestYf2:
    def test_VonMises(self):
        from mk.yieldFunction import yf2
        result = yf2.VonMises(np.array([1., 0, 0, 0, 0, 0]))
        assert len(result) == 4  # snew, phi, dphi, d2phi

    def test_Hill48(self):
        from mk.yieldFunction import yf2
        result = yf2.Hill48(np.array([1., 0, 0, 0, 0, 0]), 0.5, 0.5, 0.5, 1.5)
        assert result is not None

    def test_wrapHill48Gen(self):
        from mk.yieldFunction import yf2
        f = yf2.wrapHill48Gen(0.5, 0.5, 0.5, 1.5)
        assert callable(f)
        result = f(np.array([1., 0, 0, 0, 0, 0]))
        assert result is not None


class TestFuncFld:
    def test_import(self):
        import mk.func_fld  # noqa: F401


class TestMechtests:
    def test_inplaneTension(self, vm_func):
        import mk.tests.mechtests as mt
        result = mt.inplaneTension(vm_func)
        assert result is not None

    def test_locus(self, vm_func):
        import mk.tests.mechtests as mt
        result = mt.locus(vm_func, nth=20)
        assert result is not None
