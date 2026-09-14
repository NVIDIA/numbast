// SPDX-License-Identifier: Apache-2.0

use cuda_artifact_finalizer::{
    CudaArch, FinalizationOptions, Finalizer, FinalizerOutput, NamedInput,
};
use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel};
use serde::Deserialize;
use std::path::{Path, PathBuf};

#[allow(dead_code, unused_imports)]
mod sys {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/bindings.rs"));
}

#[cuda_module]
mod kernels {
    use super::*;

    #[kernel]
    pub fn call_generated_device_api(output: *mut i32) {
        // The public alias calls the exact `round1_add` extern in the generated
        // sys module. nvJitLink resolves it from device_api.ltoir.
        unsafe {
            *output = sys::add(20, 22);
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

fn repo_path(repo_root: &Path, configured: &str) -> PathBuf {
    let path = Path::new(configured);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        repo_root.join(path)
    }
}

fn linked_cubin() -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let example_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
    let repo_root = example_dir
        .parent()
        .and_then(Path::parent)
        .ok_or("example must remain under <repo>/examples")?;
    let manifest: BindingsManifest =
        serde_json::from_slice(&std::fs::read(example_dir.join("bindings.manifest.json"))?)?;
    let arch: CudaArch = manifest.target.gpu_arch.parse()?;
    let options = FinalizationOptions::new(arch);
    let finalizer = Finalizer::discover()?;

    let rust_ir_path = example_dir.join("numbast_cuda_oxide_round1.ll");
    let rust_ir = std::fs::read(&rust_ir_path)?;
    let rust_ltoir = finalizer.compiler().compile_nvvm_ir_to_ltoir(
        "numbast_cuda_oxide_round1.ll",
        &rust_ir,
        &options,
    )?;

    let mut inputs = vec![("numbast_cuda_oxide_round1.ltoir".to_string(), rust_ltoir)];
    for configured in manifest.artifacts.ltoir_inputs {
        let path = repo_path(repo_root, &configured);
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
    let cubin = linked_cubin()?;
    let raw_module = context.load_module_from_image(&cubin)?;
    let module = kernels::from_module(raw_module)?;
    let output = DeviceBuffer::<i32>::zeroed(&stream, 1)?;
    let output_ptr = output.cu_deviceptr() as *mut i32;

    // SAFETY: one thread writes the one-element device allocation.
    unsafe {
        module.call_generated_device_api(&stream, LaunchConfig::for_num_elems(1), output_ptr)
    }?;
    let result = output.to_host_vec(&stream)?;
    assert_eq!(result, [42]);
    println!("PASS: generated Rust called C device LTOIR directly");
    Ok(())
}
