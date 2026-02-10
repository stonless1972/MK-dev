"""Tests for mk.mk_paths"""
import numpy as np
import pytest


class TestMkPaths:
    def test_findCorrectPath(self):
        import mk.mk_paths as paths
        func, pth = paths.findCorrectPath(0)
        assert callable(func)
        assert len(pth) == 2

    def test_findCorrectPath_regions(self):
        """Test that all strain path regions are covered."""
        import mk.mk_paths as paths
        test_angles = [-30, 0, 15, 35, 50, 75, 100]
        for angle in test_angles:
            func, pth = paths.findCorrectPath(angle)
            assert callable(func)

    def test_raises_for_bad_angle(self):
        import mk.mk_paths as paths
        with pytest.raises(IOError):
            paths.findCorrectPath(999)

    def test_objf_none_uses_vm(self):
        import mk.mk_paths as paths
        result = paths.objf(0.5, None)
        assert isinstance(result, (float, np.floating))

    def test_objf_callable(self, vm_func):
        import mk.mk_paths as paths
        result = paths.objf(0.5, lambda s: vm_func(s))
        assert isinstance(result, (float, np.floating))

    def test_returnPaths(self):
        import mk.mk_paths as paths
        funcs = paths.returnPaths()
        assert len(funcs) == 6
        for f in funcs:
            assert callable(f)

    def test_each_path_returns_5_values(self):
        """Each path function should return (angs, npt, pth, stressR, stressL)."""
        import mk.mk_paths as paths
        for func in paths.returnPaths():
            result = func()
            assert len(result) == 5
