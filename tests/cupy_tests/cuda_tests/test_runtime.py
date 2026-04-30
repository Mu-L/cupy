from __future__ import annotations

import pickle
import sys

import pytest

import cupy
from cupy.cuda import driver
from cupy.cuda import nvrtc
from cupy.cuda import runtime


class TestExceptionPicklable:

    def test(self):
        e1 = runtime.CUDARuntimeError(1)
        e2 = pickle.loads(pickle.dumps(e1))
        assert e1.args == e2.args
        assert str(e1) == str(e2)


class TestMemPool:

    @pytest.mark.skipif(runtime.is_hip,
                        reason='HIP does not support async allocator')
    @pytest.mark.skipif(driver._is_cuda_python()
                        and runtime.runtimeGetVersion() < 11020,
                        reason='cudaMemPool_t is supported since CUDA 11.2')
    @pytest.mark.skipif(not driver._is_cuda_python()
                        and driver.get_build_version() < 11020,
                        reason='cudaMemPool_t is supported since CUDA 11.2')
    @pytest.mark.skipif(runtime.deviceGetAttribute(
        runtime.cudaDevAttrMemoryPoolsSupported, 0) == 0,
        reason='cudaMemPool_t is not supported on device 0')
    def test_mallocFromPoolAsync(self):
        # also test create/destroy a pool
        props = runtime.MemPoolProps(
            runtime.cudaMemAllocationTypePinned,
            runtime.cudaMemHandleTypeNone,
            runtime.cudaMemLocationTypeDevice,
            0)  # on device 0
        pool = runtime.memPoolCreate(props)
        assert pool > 0
        s = cupy.cuda.Stream()
        ptr = runtime.mallocFromPoolAsync(128, pool, s.ptr)
        assert ptr > 0
        runtime.freeAsync(ptr, s.ptr)
        runtime.memPoolDestroy(pool)


@pytest.mark.skipif(runtime.is_hip,
                    reason='This assumption is correct only in CUDA')
def test_assumed_runtime_version():
    # Verify that the CUDA runtime version returned by
    # cudaRuntimeGetVersion() is consistent with the NVRTC version
    # (both are shipped as part of the CUDA Toolkit).
    (major, minor) = nvrtc.getVersion()
    local_ver = runtime._getLocalRuntimeVersion()
    # On Windows with CUDA >= 13.0, the runtime implementation is
    # provided by the display driver (nvbugs 5955788, 5523579), so
    # cudaRuntimeGetVersion() reflects the driver's runtime version
    # rather than the toolkit version. Verify that the runtime,
    # driver, and local runtime versions all agree.
    if sys.platform == 'win32' and major >= 13:
        rt_ver = runtime.runtimeGetVersion()
        drv_ver = runtime.driverGetVersion()
        assert rt_ver == drv_ver == local_ver
    else:
        assert local_ver == major * 1000 + minor * 10


def test_major_version():
    major = runtime._getCUDAMajorVersion()
    if runtime.is_hip:
        assert major == 0
    else:
        assert 10 < major < 20
