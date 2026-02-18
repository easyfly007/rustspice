use sim_va::OsdiLibrary;
use std::path::Path;

#[test]
fn load_nonexistent_file_returns_error() {
    let result = OsdiLibrary::load(Path::new("/nonexistent/model.osdi"));
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected error for nonexistent file"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("failed to load") || msg.contains("not found") || msg.contains("No such file"),
        "Unexpected error: {}",
        msg
    );
}

#[test]
fn load_non_shared_library_returns_error() {
    // Create a regular file that is not a valid shared library
    let dir = std::env::temp_dir().join("rustspice_test_loader");
    std::fs::create_dir_all(&dir).unwrap();
    let fake_osdi = dir.join("not_a_lib.osdi");
    std::fs::write(&fake_osdi, b"this is not a shared library").unwrap();

    let result = OsdiLibrary::load(&fake_osdi);
    let err = match result {
        Err(e) => e,
        Ok(_) => panic!("expected error for invalid shared library"),
    };
    let msg = err.to_string();
    assert!(
        msg.contains("failed to load") || msg.contains("invalid"),
        "Unexpected error: {}",
        msg
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn load_empty_file_returns_error() {
    let dir = std::env::temp_dir().join("rustspice_test_loader_empty");
    std::fs::create_dir_all(&dir).unwrap();
    let empty_osdi = dir.join("empty.osdi");
    std::fs::write(&empty_osdi, b"").unwrap();

    let result = OsdiLibrary::load(&empty_osdi);
    assert!(result.is_err());

    let _ = std::fs::remove_dir_all(&dir);
}
