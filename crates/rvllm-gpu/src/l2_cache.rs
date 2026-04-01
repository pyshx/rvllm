//! L2 cache persistence control for SM 8.0+ (Ampere/Hopper/Blackwell).
//!
//! Configures the device-level L2 set-aside and per-stream access policy
//! windows so that small, frequently-accessed buffers (KV cache pages,
//! RoPE tables, norm weights) stay resident in L2 across kernel launches.

use std::sync::Arc;
use cudarc::driver::CudaStream;
use cudarc::driver::sys;

/// Reserve a fraction of L2 for persisting accesses (device-level, once at init).
/// H100 has 50 MB L2; reserving 75% = 37.5 MB for persisting data.
pub fn configure_l2_persisting_cache(fraction: f64) -> Result<(), String> {
    let mut prop = std::mem::MaybeUninit::<sys::CUdevprop_v1>::uninit();
    // Use cuDeviceGetAttribute for l2CacheSize and persistingL2CacheMaxSize
    let mut l2_size: i32 = 0;
    let mut max_persist: i32 = 0;
    unsafe {
        sys::cuDeviceGetAttribute(
            &mut l2_size,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE,
            sys::CUdevice_v1(0),
        ).result().map_err(|e| format!("query L2 size: {e}"))?;

        sys::cuDeviceGetAttribute(
            &mut max_persist,
            sys::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
            sys::CUdevice_v1(0),
        ).result().map_err(|e| format!("query max persist L2: {e}"))?;
    }

    if l2_size == 0 || max_persist == 0 {
        tracing::debug!(l2_size, max_persist, "L2 persistence not supported, skipping");
        return Ok(());
    }

    let desired = ((l2_size as f64) * fraction) as usize;
    let capped = desired.min(max_persist as usize);

    unsafe {
        sys::cuCtxSetLimit(
            sys::CUlimit_enum::CU_LIMIT_PERSISTING_L2_CACHE_SIZE,
            capped,
        ).result().map_err(|e| format!("set L2 persist size: {e}"))?;
    }

    tracing::info!(
        l2_total_mb = l2_size as f64 / 1048576.0,
        persist_mb = capped as f64 / 1048576.0,
        "L2 persisting cache configured"
    );
    Ok(())
}

/// Set stream-level access policy window: accesses to [ptr, ptr+num_bytes)
/// will be marked as persisting in L2.
///
/// # Safety
/// `ptr` must be a valid device pointer and the memory must remain allocated
/// for the lifetime of the stream attribute.
pub unsafe fn set_stream_l2_persist(
    stream: &Arc<CudaStream>,
    ptr: u64,
    num_bytes: usize,
    hit_ratio: f32,
) -> Result<(), String> {
    if num_bytes == 0 {
        return Ok(());
    }

    let window = sys::CUaccessPolicyWindow_st {
        base_ptr: ptr as *mut std::ffi::c_void,
        num_bytes,
        hitRatio: hit_ratio,
        hitProp: sys::CUaccessProperty_enum::CU_ACCESS_PROPERTY_PERSISTING,
        missProp: sys::CUaccessProperty_enum::CU_ACCESS_PROPERTY_STREAMING,
    };

    let mut attr_value: sys::CUstreamAttrValue_union = std::mem::zeroed();
    attr_value.accessPolicyWindow = window;

    sys::cuStreamSetAttribute(
        stream.cu_stream(),
        sys::CUstreamAttrID_enum::CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
        &attr_value as *const sys::CUstreamAttrValue_union as *const _,
    ).result().map_err(|e| format!("set stream L2 policy: {e}"))?;

    tracing::debug!(
        ptr, num_bytes,
        hit_ratio,
        "stream L2 access policy window set"
    );
    Ok(())
}

/// Clear the stream-level access policy window (reset to normal caching).
pub fn clear_stream_l2_persist(stream: &Arc<CudaStream>) -> Result<(), String> {
    unsafe {
        let window = sys::CUaccessPolicyWindow_st {
            base_ptr: std::ptr::null_mut(),
            num_bytes: 0,
            hitRatio: 0.0,
            hitProp: sys::CUaccessProperty_enum::CU_ACCESS_PROPERTY_NORMAL,
            missProp: sys::CUaccessProperty_enum::CU_ACCESS_PROPERTY_NORMAL,
        };

        let mut attr_value: sys::CUstreamAttrValue_union = std::mem::zeroed();
        attr_value.accessPolicyWindow = window;

        sys::cuStreamSetAttribute(
            stream.cu_stream(),
            sys::CUstreamAttrID_enum::CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            &attr_value as *const sys::CUstreamAttrValue_union as *const _,
        ).result().map_err(|e| format!("clear stream L2 policy: {e}"))?;
    }
    Ok(())
}
