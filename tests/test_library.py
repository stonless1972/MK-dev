"""Tests for mk.library.lib and mk.library.mk_lib"""
import numpy as np


class TestLib:
    def test_convert_6sig_princ(self):
        import mk.library.lib as lib
        result = lib.convert_6sig_princ(np.array([1., 0, 0, 0, 0, 0]))
        assert result is not None

    def test_convert_sig33_sig6(self):
        import mk.library.lib as lib
        result = lib.convert_sig33_sig6(np.eye(3))
        assert len(result) == 6

    def test_convert_sig6_sig33(self):
        import mk.library.lib as lib
        result = lib.convert_sig6_sig33(np.array([1., 0, 0, 0, 0, 0]))
        assert result.shape == (3, 3)

    def test_gen_tempfile(self):
        import mk.library.lib as lib
        result = lib.gen_tempfile(ext="log")
        assert isinstance(result, str)

    def test_rhos2ths(self):
        import mk.library.lib as lib
        result = lib.rhos2ths(np.array([0., 0.5, 1.]))
        assert result is not None

    def test_rot_6d(self):
        import mk.library.lib as lib
        result = lib.rot_6d(np.array([1., 0, 0, 0, 0, 0]), 0.5)
        assert len(result) == 6

    def test_draw_guide(self):
        import mk.library.lib as lib
        mock_ax = type("ax", (), {
            "get_xlim": lambda s: (0, 1),
            "get_ylim": lambda s: (0, 1),
            "plot": lambda s, *a, **k: None,
            "set_xlim": lambda s, *a: None,
            "set_ylim": lambda s, *a: None,
        })()
        lib.draw_guide(mock_ax)


class TestMkLib:
    def test_findStressOnYS(self, vm_func):
        import mk.library.mk_lib as mk_lib
        s, dphi = mk_lib.findStressOnYS(
            vm_func,
            np.array([1., 0, 0, 0, 0, 0.]),
            np.array([0., 1., 0, 0, 0, 0.]),
            pth=[1., 0.5], verbose=False)
        assert s is not None
        assert dphi is not None
