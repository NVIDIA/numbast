// SPDX-License-Identifier: Apache-2.0

#![feature(f16)]
#![allow(unused_features)]

use cuda_artifact_finalizer::{
    CudaArch, FinalizationOptions, Finalizer, FinalizerOutput, NamedInput,
};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[allow(
    dead_code,
    non_camel_case_types,
    non_upper_case_globals,
    unused_imports
)]
mod nvshmem_device {
    include!(env!("NUMBAST_NVSHMEM_RUST_BINDINGS"));
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn query_nvshmem_version(output: *mut i32) {
        unsafe {
            nvshmem_device::vendor_get_version_info(output, output.add(1), output.add(2));
        }
    }
}

#[derive(Deserialize)]
struct Target {
    gpu_arch: String,
}

#[derive(Deserialize)]
struct Artifacts {
    ltoir_inputs: Vec<String>,
}

#[derive(Deserialize)]
struct BindingsManifest {
    target: Target,
    artifacts: Artifacts,
}

fn configured_path(base: &Path, configured: &str) -> PathBuf {
    let path = Path::new(configured);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        base.join(path)
    }
}

fn linked_cubin() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let crate_root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let config_base = Path::new(env!("NUMBAST_CONFIG_BASE"));
    let manifest: BindingsManifest =
        serde_json::from_slice(&std::fs::read(env!("NUMBAST_NVSHMEM_MANIFEST"))?)?;
    let arch: CudaArch = manifest.target.gpu_arch.parse()?;
    let options = FinalizationOptions::new(arch);
    let finalizer = Finalizer::discover()?;

    let rust_ir_path = crate_root.join("numbast_nvshmem_cuda_oxide_e2e.ll");
    let rust_ir = std::fs::read(&rust_ir_path)?;
    let rust_ltoir = finalizer.compiler().compile_nvvm_ir_to_ltoir(
        "numbast_nvshmem_cuda_oxide_e2e.ll",
        &rust_ir,
        &options,
    )?;

    let mut inputs = vec![(
        "numbast_nvshmem_cuda_oxide_e2e.ltoir".to_string(),
        rust_ltoir,
    )];
    for configured in manifest.artifacts.ltoir_inputs {
        let path = configured_path(config_base, &configured);
        inputs.push((configured, std::fs::read(path)?));
    }
    let named_inputs: Vec<_> = inputs
        .iter()
        .map(|(name, bytes)| NamedInput::new(name, bytes))
        .collect();
    Ok(finalizer.link_ltoir(&named_inputs, &options, FinalizerOutput::Cubin)?)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let context = CudaContext::new(0)?;
    let stream = context.default_stream();
    let raw_module = context.load_module_from_image(&linked_cubin()?)?;
    let module = kernels::from_module(raw_module)?;
    let output = DeviceBuffer::<i32>::zeroed(&stream, 3)?;

    unsafe {
        module.query_nvshmem_version(
            &stream,
            LaunchConfig::for_num_elems(1),
            output.cu_deviceptr() as *mut i32,
        )
    }?;
    let version = output.to_host_vec(&stream)?;
    assert!(version[0] > 0, "invalid NVSHMEM version: {version:?}");
    println!(
        "PASS: generated NVSHMEM binding linked and ran from LTOIR ({}.{}.{})",
        version[0], version[1], version[2]
    );
    Ok(())
}
