use async_trait::async_trait;
use ds_agent_core::{
    AgentTool, BoundedText, DuplicateToolError, ToolExecutionContext, ToolExecutor, ToolOutput,
    ToolRegistry,
};
use ds_ai::Tool;
use serde::Deserialize;
use serde_json::json;
use similar::TextDiff;
use std::{
    ffi::OsStr,
    io,
    path::{Path, PathBuf},
    process::Stdio,
    time::Duration,
};
use tokio::{
    io::AsyncReadExt,
    process::{Child, Command},
};

const MAX_LINES: usize = 2_000;
const MAX_BYTES: usize = 50 * 1_024;
const DEFAULT_COMMAND_TIMEOUT_SECONDS: u64 = 120;
const MAX_DIFF_BYTES: usize = 16 * 1_024;

pub fn coding_tools() -> Result<ToolRegistry, DuplicateToolError> {
    ToolRegistry::new([read_tool(), bash_tool(), edit_tool(), write_tool()])
}

fn read_tool() -> AgentTool {
    AgentTool::new(
        Tool::new(
            "read",
            "Read a UTF-8 text file. Offset is one-based. Output is limited to 2,000 lines or 50 KiB.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "minLength": 1 },
                    "offset": { "type": "integer", "minimum": 1 },
                    "limit": { "type": "integer", "minimum": 1 }
                },
                "required": ["path"],
                "additionalProperties": false
            }),
        ),
        ReadTool,
    )
}

fn bash_tool() -> AgentTool {
    AgentTool::new(
        Tool::new(
            "bash",
            "Run a shell command in the current working directory. Output is limited to the final 2,000 lines or 50 KiB.",
            json!({
                "type": "object",
                "properties": {
                    "command": { "type": "string", "minLength": 1 },
                    "timeout": { "type": "integer", "minimum": 1, "maximum": 3600 }
                },
                "required": ["command"],
                "additionalProperties": false
            }),
        ),
        BashTool,
    )
}

fn edit_tool() -> AgentTool {
    AgentTool::new(
        Tool::new(
            "edit",
            "Replace exact text in a UTF-8 file. Every oldText must occur exactly once and all edits are validated before writing.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "minLength": 1 },
                    "edits": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "properties": {
                                "oldText": { "type": "string", "minLength": 1 },
                                "newText": { "type": "string" }
                            },
                            "required": ["oldText", "newText"],
                            "additionalProperties": false
                        }
                    }
                },
                "required": ["path", "edits"],
                "additionalProperties": false
            }),
        ),
        EditTool,
    )
}

fn write_tool() -> AgentTool {
    AgentTool::new(
        Tool::new(
            "write",
            "Create or overwrite a UTF-8 file, creating parent directories when needed.",
            json!({
                "type": "object",
                "properties": {
                    "path": { "type": "string", "minLength": 1 },
                    "content": { "type": "string" }
                },
                "required": ["path", "content"],
                "additionalProperties": false
            }),
        ),
        WriteTool,
    )
}

#[derive(Deserialize)]
struct ReadArguments {
    path: String,
    offset: Option<usize>,
    limit: Option<usize>,
}

struct ReadTool;

#[async_trait]
impl ToolExecutor for ReadTool {
    async fn execute(
        &self,
        arguments: serde_json::Value,
        context: ToolExecutionContext<'_>,
    ) -> ToolOutput {
        let arguments = match serde_json::from_value::<ReadArguments>(arguments) {
            Ok(arguments) => arguments,
            Err(error) => return ToolOutput::error(format!("invalid read arguments: {error}")),
        };
        let offset = arguments.offset.unwrap_or(1);
        if offset == 0 {
            return ToolOutput::error("read offset must be at least 1");
        }
        let limit = arguments.limit.unwrap_or(MAX_LINES).min(MAX_LINES);
        if limit == 0 {
            return ToolOutput::error("read limit must be at least 1");
        }
        let path = resolve_path(context.working_directory, &arguments.path);
        let read = tokio::fs::read_to_string(&path);
        tokio::pin!(read);
        let contents = tokio::select! {
            contents = &mut read => match contents {
                Ok(contents) => contents,
                Err(error) => return ToolOutput::error(format!("failed to read {}: {error}", path.display())),
            },
            () = context.cancellation.cancelled() => return ToolOutput::error("read cancelled"),
        };
        ToolOutput::success(read_slice(&contents, offset, limit))
    }
}

#[derive(Deserialize)]
struct BashArguments {
    command: String,
    timeout: Option<u64>,
}

struct BashTool;

#[async_trait]
impl ToolExecutor for BashTool {
    async fn execute(
        &self,
        arguments: serde_json::Value,
        context: ToolExecutionContext<'_>,
    ) -> ToolOutput {
        let arguments = match serde_json::from_value::<BashArguments>(arguments) {
            Ok(arguments) => arguments,
            Err(error) => return ToolOutput::error(format!("invalid bash arguments: {error}")),
        };
        let timeout =
            Duration::from_secs(arguments.timeout.unwrap_or(DEFAULT_COMMAND_TIMEOUT_SECONDS));
        run_command(
            &arguments.command,
            timeout,
            context.working_directory,
            context.cancellation,
        )
        .await
    }
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct EditOperation {
    old_text: String,
    new_text: String,
}

#[derive(Deserialize)]
struct EditArguments {
    path: String,
    edits: Vec<EditOperation>,
}

struct EditTool;

#[async_trait]
impl ToolExecutor for EditTool {
    async fn execute(
        &self,
        arguments: serde_json::Value,
        context: ToolExecutionContext<'_>,
    ) -> ToolOutput {
        let arguments = match serde_json::from_value::<EditArguments>(arguments) {
            Ok(arguments) => arguments,
            Err(error) => return ToolOutput::error(format!("invalid edit arguments: {error}")),
        };
        if arguments.edits.is_empty() {
            return ToolOutput::error("edit requires at least one replacement");
        }
        let path = resolve_path(context.working_directory, &arguments.path);
        let contents = match cancellable_read(&path, context.cancellation).await {
            Ok(contents) => contents,
            Err(error) => {
                return ToolOutput::error(format!("failed to read {}: {error}", path.display()));
            }
        };
        let edited = match apply_edits(&contents, &arguments.edits) {
            Ok(edited) => edited,
            Err(error) => return ToolOutput::error(error),
        };
        if context.cancellation.is_cancelled() {
            return ToolOutput::error("edit cancelled");
        }
        if let Err(error) = tokio::fs::write(&path, &edited).await {
            return ToolOutput::error(format!("failed to write {}: {error}", path.display()));
        }
        let (diff, diff_truncated) = render_diff(&contents, &edited, &arguments.path);
        ToolOutput::success(BoundedText::new(
            format!("edited {}", path.display()),
            false,
        ))
        .with_details(json!({
            "path": arguments.path,
            "diff": diff,
            "diff_truncated": diff_truncated
        }))
    }
}

#[derive(Deserialize)]
struct WriteArguments {
    path: String,
    content: String,
}

struct WriteTool;

#[async_trait]
impl ToolExecutor for WriteTool {
    async fn execute(
        &self,
        arguments: serde_json::Value,
        context: ToolExecutionContext<'_>,
    ) -> ToolOutput {
        let arguments = match serde_json::from_value::<WriteArguments>(arguments) {
            Ok(arguments) => arguments,
            Err(error) => return ToolOutput::error(format!("invalid write arguments: {error}")),
        };
        let path = resolve_path(context.working_directory, &arguments.path);
        if let Some(parent) = path.parent()
            && let Err(error) = tokio::fs::create_dir_all(parent).await
        {
            return ToolOutput::error(format!("failed to create {}: {error}", parent.display()));
        }
        if context.cancellation.is_cancelled() {
            return ToolOutput::error("write cancelled");
        }
        if let Err(error) = tokio::fs::write(&path, arguments.content).await {
            return ToolOutput::error(format!("failed to write {}: {error}", path.display()));
        }
        ToolOutput::success(BoundedText::new(format!("wrote {}", path.display()), false))
    }
}

fn resolve_path(working_directory: &Path, path: &str) -> PathBuf {
    let path = Path::new(path);
    if path.is_absolute() {
        path.to_path_buf()
    } else {
        working_directory.join(path)
    }
}

fn read_slice(contents: &str, offset: usize, limit: usize) -> BoundedText {
    let lines = contents.split_inclusive('\n').collect::<Vec<_>>();
    let start = (offset - 1).min(lines.len());
    let end = (start + limit).min(lines.len());
    let selected = lines[start..end].concat();
    let line_truncated = end < lines.len();
    if !line_truncated && selected.len() <= MAX_BYTES {
        return BoundedText::new(selected, false);
    }

    let hint_reserve = 80.min(MAX_BYTES);
    let mut body = truncate_utf8_head(&selected, MAX_BYTES - hint_reserve);
    let displayed_lines = body
        .as_bytes()
        .iter()
        .filter(|byte| **byte == b'\n')
        .count()
        + usize::from(!body.is_empty() && !body.ends_with('\n'));
    let next_offset = offset + displayed_lines;
    let hint = format!("\n[truncated; continue with offset {next_offset}]\n");
    body.truncate(fitting_boundary(&body, MAX_BYTES - hint.len()));
    body.push_str(&hint);
    BoundedText::new(body, true)
}

fn truncate_utf8_head(value: &str, max_bytes: usize) -> String {
    value[..fitting_boundary(value, max_bytes)].to_owned()
}

fn fitting_boundary(value: &str, max_bytes: usize) -> usize {
    let mut index = value.len().min(max_bytes);
    while !value.is_char_boundary(index) {
        index -= 1;
    }
    index
}

fn bounded_tail(value: &str, max_bytes: usize) -> BoundedText {
    let lines = value.split_inclusive('\n').collect::<Vec<_>>();
    let start = lines.len().saturating_sub(MAX_LINES);
    let line_truncated = start > 0;
    let selected = lines[start..].concat();
    if !line_truncated && selected.len() <= max_bytes {
        return BoundedText::new(selected, false);
    }
    let marker = "[earlier output truncated]\n";
    let budget = max_bytes.saturating_sub(marker.len());
    let mut start = selected.len().saturating_sub(budget);
    while !selected.is_char_boundary(start) {
        start += 1;
    }
    let mut text = String::with_capacity(max_bytes);
    text.push_str(marker);
    text.push_str(&selected[start..]);
    BoundedText::new(text, true)
}

async fn cancellable_read(
    path: &Path,
    cancellation: &tokio_util::sync::CancellationToken,
) -> io::Result<String> {
    tokio::select! {
        result = tokio::fs::read_to_string(path) => result,
        () = cancellation.cancelled() => Err(io::Error::new(io::ErrorKind::Interrupted, "cancelled")),
    }
}

fn apply_edits(contents: &str, edits: &[EditOperation]) -> Result<String, String> {
    let (bom, body) = contents
        .strip_prefix('\u{feff}')
        .map_or(("", contents), |body| ("\u{feff}", body));
    let crlf = body.contains("\r\n");
    let mut replacements = Vec::with_capacity(edits.len());
    for edit in edits {
        let old_text = normalize_line_endings(&edit.old_text, crlf);
        if old_text.is_empty() {
            return Err("oldText must not be empty".into());
        }
        let matches = body.match_indices(&old_text).collect::<Vec<_>>();
        if matches.len() != 1 {
            return Err(format!(
                "oldText must match exactly once; found {} matches",
                matches.len()
            ));
        }
        let start = matches[0].0;
        replacements.push((
            start..start + old_text.len(),
            normalize_line_endings(&edit.new_text, crlf),
        ));
    }
    replacements.sort_by_key(|(range, _)| range.start);
    if replacements
        .windows(2)
        .any(|pair| pair[0].0.end > pair[1].0.start)
    {
        return Err("edit replacements overlap".into());
    }

    let mut result = body.to_owned();
    for (range, replacement) in replacements.into_iter().rev() {
        result.replace_range(range, &replacement);
    }
    Ok(format!("{bom}{result}"))
}

fn normalize_line_endings(value: &str, crlf: bool) -> String {
    let normalized = value.replace("\r\n", "\n");
    if crlf {
        normalized.replace('\n', "\r\n")
    } else {
        normalized
    }
}

fn render_diff(before: &str, after: &str, path: &str) -> (String, bool) {
    let diff = TextDiff::from_lines(before, after)
        .unified_diff()
        .header(&format!("a/{path}"), &format!("b/{path}"))
        .to_string();
    if diff.len() <= MAX_DIFF_BYTES {
        return (diff, false);
    }
    let mut end = MAX_DIFF_BYTES;
    while !diff.is_char_boundary(end) {
        end -= 1;
    }
    (diff[..end].to_owned(), true)
}

async fn run_command(
    script: &str,
    timeout: Duration,
    working_directory: &Path,
    cancellation: &tokio_util::sync::CancellationToken,
) -> ToolOutput {
    let shell = std::env::var_os("SHELL").unwrap_or_else(|| "/bin/sh".into());
    let mut command = shell_command(&shell, script, working_directory);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => return ToolOutput::error(format!("failed to start command: {error}")),
    };
    let stdout = child.stdout.take().expect("configured stdout pipe");
    let stderr = child.stderr.take().expect("configured stderr pipe");
    let stdout = tokio::spawn(read_pipe(stdout));
    let stderr = tokio::spawn(read_pipe(stderr));
    let result = tokio::select! {
        status = child.wait() => CommandResult::Exited(status),
        () = tokio::time::sleep(timeout) => {
            terminate(&mut child).await;
            CommandResult::TimedOut
        }
        () = cancellation.cancelled() => {
            terminate(&mut child).await;
            CommandResult::Cancelled
        }
    };
    let output = match collect_output(stdout, stderr).await {
        Ok(output) => bounded_tail(&output, MAX_BYTES),
        Err(error) => {
            return ToolOutput::error(format!("failed to capture command output: {error}"));
        }
    };
    match result {
        CommandResult::Exited(Ok(status)) if status.success() => {
            ToolOutput::success(output).with_details(json!({ "exit_code": status.code() }))
        }
        CommandResult::Exited(Ok(status)) => ToolOutput {
            content: prefix_bounded(format!("command exited with {status}"), output),
            is_error: true,
            details: Some(json!({ "exit_code": status.code() })),
        },
        CommandResult::Exited(Err(error)) => ToolOutput::error(format!("command failed: {error}"))
            .with_details(json!({ "exit_code": null })),
        CommandResult::TimedOut => ToolOutput {
            content: prefix_bounded(
                format!("command timed out after {}s", timeout.as_secs()),
                output,
            ),
            is_error: true,
            details: Some(json!({ "timed_out": true })),
        },
        CommandResult::Cancelled => ToolOutput {
            content: prefix_bounded("command cancelled".into(), output),
            is_error: true,
            details: Some(json!({ "cancelled": true })),
        },
    }
}

fn shell_command(shell: &OsStr, script: &str, working_directory: &Path) -> Command {
    let mut command = Command::new(shell);
    command
        .arg("-lc")
        .arg(script)
        .current_dir(working_directory)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt as _;
        command.as_std_mut().process_group(0);
    }
    command
}

enum CommandResult {
    Exited(io::Result<std::process::ExitStatus>),
    TimedOut,
    Cancelled,
}

async fn read_pipe(mut pipe: impl tokio::io::AsyncRead + Unpin) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    pipe.read_to_end(&mut bytes).await?;
    Ok(bytes)
}

async fn collect_output(
    stdout: tokio::task::JoinHandle<io::Result<Vec<u8>>>,
    stderr: tokio::task::JoinHandle<io::Result<Vec<u8>>>,
) -> io::Result<String> {
    let stdout = stdout.await.map_err(io::Error::other)??;
    let stderr = stderr.await.map_err(io::Error::other)??;
    let mut output = String::from_utf8_lossy(&stdout).into_owned();
    output.push_str(&String::from_utf8_lossy(&stderr));
    Ok(output)
}

fn prefix_bounded(prefix: String, output: BoundedText) -> BoundedText {
    let separator = if output.text.is_empty() { "" } else { "\n" };
    let available = MAX_BYTES.saturating_sub(prefix.len() + separator.len());
    let bounded = bounded_tail(&output.text, available);
    BoundedText::new(
        format!("{prefix}{separator}{}", bounded.text),
        output.truncated || bounded.truncated,
    )
}

async fn terminate(child: &mut Child) {
    #[cfg(unix)]
    if let Some(id) = child.id()
        && let Ok(id) = i32::try_from(id)
    {
        let _ = nix::sys::signal::killpg(
            nix::unistd::Pid::from_raw(id),
            nix::sys::signal::Signal::SIGKILL,
        );
    }
    let _ = child.kill().await;
    let _ = child.wait().await;
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;
    use tokio_util::sync::CancellationToken;

    fn context<'a>(
        directory: &'a Path,
        cancellation: &'a CancellationToken,
    ) -> ToolExecutionContext<'a> {
        ToolExecutionContext {
            working_directory: directory,
            cancellation,
        }
    }

    #[tokio::test]
    async fn read_honors_offsets_and_limits() {
        let directory = tempdir().unwrap();
        let content = (1..=2_100)
            .map(|line| format!("line {line}\n"))
            .collect::<String>();
        tokio::fs::write(directory.path().join("file.txt"), content)
            .await
            .unwrap();
        let cancellation = CancellationToken::new();

        let output = ReadTool
            .execute(
                json!({ "path": "file.txt", "offset": 2, "limit": 2 }),
                context(directory.path(), &cancellation),
            )
            .await;
        assert!(output.content.text.starts_with("line 2\nline 3\n"));
        assert!(output.content.text.contains("continue with offset 4"));

        let output = ReadTool
            .execute(
                json!({ "path": "file.txt" }),
                context(directory.path(), &cancellation),
            )
            .await;
        assert!(output.content.truncated);
        assert!(output.content.text.len() <= MAX_BYTES);
        assert!(output.content.text.contains("continue with offset"));
    }

    #[tokio::test]
    async fn edit_is_exact_and_all_or_nothing() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("file.txt");
        tokio::fs::write(&path, "one two two three\n")
            .await
            .unwrap();
        let cancellation = CancellationToken::new();

        let output = EditTool
            .execute(
                json!({
                    "path": "file.txt",
                    "edits": [
                        { "oldText": "one", "newText": "ONE" },
                        { "oldText": "two", "newText": "TWO" }
                    ]
                }),
                context(directory.path(), &cancellation),
            )
            .await;

        assert!(output.is_error);
        assert_eq!(
            tokio::fs::read_to_string(&path).await.unwrap(),
            "one two two three\n"
        );
    }

    #[tokio::test]
    async fn edit_preserves_bom_and_crlf() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("file.txt");
        tokio::fs::write(&path, "\u{feff}one\r\ntwo\r\n")
            .await
            .unwrap();
        let cancellation = CancellationToken::new();

        let output = EditTool
            .execute(
                json!({
                    "path": "file.txt",
                    "edits": [{ "oldText": "one\ntwo", "newText": "ONE\nTWO" }]
                }),
                context(directory.path(), &cancellation),
            )
            .await;

        assert!(!output.is_error);
        assert_eq!(
            tokio::fs::read_to_string(&path).await.unwrap(),
            "\u{feff}ONE\r\nTWO\r\n"
        );
    }

    #[tokio::test]
    async fn edit_reports_a_bounded_unified_diff() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("file.txt");
        tokio::fs::write(&path, "one\ntwo\n").await.unwrap();
        let cancellation = CancellationToken::new();

        let output = EditTool
            .execute(
                json!({
                    "path": "file.txt",
                    "edits": [{ "oldText": "two", "newText": "TWO" }]
                }),
                context(directory.path(), &cancellation),
            )
            .await;

        let details = output.details.unwrap();
        let diff = details["diff"].as_str().unwrap();
        assert!(diff.contains("--- a/file.txt"));
        assert!(diff.contains("+++ b/file.txt"));
        assert!(diff.contains("-two"));
        assert!(diff.contains("+TWO"));
        assert_eq!(details["diff_truncated"], false);
    }

    #[tokio::test]
    async fn write_creates_parents_and_overwrites() {
        let directory = tempdir().unwrap();
        let cancellation = CancellationToken::new();
        for content in ["first", "second"] {
            let output = WriteTool
                .execute(
                    json!({ "path": "nested/file.txt", "content": content }),
                    context(directory.path(), &cancellation),
                )
                .await;
            assert!(!output.is_error);
        }
        assert_eq!(
            tokio::fs::read_to_string(directory.path().join("nested/file.txt"))
                .await
                .unwrap(),
            "second"
        );
    }

    #[tokio::test]
    async fn bash_captures_output_and_nonzero_status() {
        let directory = tempdir().unwrap();
        let cancellation = CancellationToken::new();
        let success = BashTool
            .execute(
                json!({ "command": "printf out; printf err >&2" }),
                context(directory.path(), &cancellation),
            )
            .await;
        assert!(!success.is_error);
        assert!(success.content.text.contains("out"));
        assert!(success.content.text.contains("err"));
        assert_eq!(success.details, Some(json!({ "exit_code": 0 })));

        let failure = BashTool
            .execute(
                json!({ "command": "printf nope; exit 7" }),
                context(directory.path(), &cancellation),
            )
            .await;
        assert!(failure.is_error);
        assert!(failure.content.text.contains("status: 7"));
        assert!(failure.content.text.contains("nope"));
        assert_eq!(failure.details, Some(json!({ "exit_code": 7 })));
    }

    #[test]
    fn bash_script_is_not_prefixed_with_shell_specific_exec_syntax() {
        let command = shell_command(OsStr::new("/bin/sh"), "printf alpha", Path::new("/tmp"));
        let arguments = command
            .as_std()
            .get_args()
            .map(|argument| argument.to_string_lossy().into_owned())
            .collect::<Vec<_>>();

        assert_eq!(arguments, ["-lc", "printf alpha"]);
    }

    #[tokio::test]
    async fn bash_times_out_and_cancels() {
        let directory = tempdir().unwrap();
        let cancellation = CancellationToken::new();
        let timed_out = BashTool
            .execute(
                json!({ "command": "sleep 10", "timeout": 1 }),
                context(directory.path(), &cancellation),
            )
            .await;
        assert!(timed_out.is_error);
        assert!(timed_out.content.text.contains("timed out"));
        assert_eq!(timed_out.details, Some(json!({ "timed_out": true })));

        let cancellation = CancellationToken::new();
        let trigger = cancellation.clone();
        tokio::spawn(async move {
            tokio::time::sleep(Duration::from_millis(50)).await;
            trigger.cancel();
        });
        let cancelled = BashTool
            .execute(
                json!({ "command": "sleep 10" }),
                context(directory.path(), &cancellation),
            )
            .await;
        assert!(cancelled.is_error);
        assert!(cancelled.content.text.contains("cancelled"));
        assert_eq!(cancelled.details, Some(json!({ "cancelled": true })));
    }

    #[test]
    fn failing_command_prefix_survives_large_output() {
        let output = BoundedText::new("x".repeat(MAX_BYTES * 2), false);

        let bounded = prefix_bounded("command exited with status: 7".into(), output);

        assert!(bounded.text.starts_with("command exited with status: 7\n"));
        assert!(bounded.text.len() <= MAX_BYTES);
        assert!(bounded.truncated);
    }
}
