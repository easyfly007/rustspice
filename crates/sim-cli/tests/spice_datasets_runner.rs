use std::path::PathBuf;
use std::process::Command;

#[test]
fn run_spice_datasets_script() {
    let repo_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..");
    let script = repo_root.join("tests").join("run_spice_datasets.py");

    let status = Command::new("python")
        .arg(script)
        .current_dir(&repo_root)
        .status()
        .expect("failed to run spice-datasets script");

    assert!(status.success(), "spice-datasets script failed");
}
