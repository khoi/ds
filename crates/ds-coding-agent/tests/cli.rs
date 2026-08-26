use std::process::Command;

#[test]
fn help_finishes_without_terminal_initialization() {
    let output = Command::new(env!("CARGO_BIN_EXE_ds"))
        .arg("--help")
        .output()
        .unwrap();

    assert!(output.status.success());
    assert!(output.stderr.is_empty());
    assert!(!output.stdout.windows(2).any(|bytes| bytes == b"\x1b["));
    let help = String::from_utf8(output.stdout).unwrap();
    assert!(help.contains("--model <PROVIDER/MODEL>"));
}
