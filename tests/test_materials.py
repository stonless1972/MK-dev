"""Tests for mk.materials modules"""
import numpy as np
import pytest


class TestConstitutive:
    def test_snapshot_takeshot(self):
        import mk.materials.constitutive as const
        snap = const.Snapshot()
        snap.takeshot(x=1., y=2.)

    def test_set_hrd_raises_for_bad_label(self):
        import mk.materials.constitutive as const
        c = const.Constitutive()
        c.label_hrd = "bad"
        c.params_hrd = [1, 2, 3, 4]
        with pytest.raises(IOError):
            c.set_hrd()

    def test_recordCurrentStat(self):
        import mk.library.lib as lib
        import mk.materials.constitutive as const
        c = const.Constitutive()
        c.eps = 0.1
        c.sig = 100.
        c.stress = np.zeros(6)
        c.dphi = np.zeros(6)
        c.logfn = lib.gen_tempfile(prefix="test", ext="log")
        c.recordCurrentStat()


class TestFuncSr:
    def test_power_law(self):
        import mk.materials.func_sr as fsr
        f = fsr.c_F(0, ed0=1., m=0.5)
        result = f(2.0)
        assert result > 0

    def test_raises_for_bad_iopt(self):
        import mk.materials.func_sr as fsr
        f = fsr.c_F(99)
        with pytest.raises(IOError):
            f(1.0)

    def test_func_jc(self):
        import mk.materials.func_sr as fsr
        result = fsr.func_jc(2.0, ed0=1.0, m=0.05)
        assert result > 0


class TestFuncHardFor:
    def test_return_swift(self):
        import mk.materials.func_hard_for as fhf
        f = fhf.return_swift(n=0.25, m=0.05, ks=600., e0=0.0004, qq=1e3)
        result = f(0.1)
        assert result is not None

    def test_return_voce(self):
        import mk.materials.func_hard_for as fhf
        f = fhf.return_voce(a=479., b0=340., c=7.68, b1=70.9, m=0.05, qq=1e3)
        result = f(0.1)
        assert result is not None


class TestMaterials:
    def test_raises_for_string_input(self):
        import mk.materials.materials as mat
        with pytest.raises(IOError):
            mat.library("bad")

    def test_raises_for_bad_iopt(self):
        import mk.materials.materials as mat
        with pytest.raises(IOError):
            mat.library(999)
