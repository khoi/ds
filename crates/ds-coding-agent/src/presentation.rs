use ds_agent_core::ToolOutput;
use serde_json::Value;
use std::time::Duration;

const MAX_LABEL_CHARS: usize = 120;
const MAX_OUTPUT_LINES: usize = 5;
const MAX_DIFF_LINES: usize = 12;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum PresentationOutcome {
    Success,
    Error,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ToolPresentation {
    pub headline: String,
    pub output: Vec<String>,
    pub outcome: PresentationOutcome,
}

pub(crate) fn active_tool(name: &str, arguments: &Value) -> String {
    action_label(name, arguments, false)
}

pub(crate) fn finished_tool(
    name: &str,
    arguments: &Value,
    output: &ToolOutput,
    duration: Duration,
) -> ToolPresentation {
    let outcome = if output.is_error {
        PresentationOutcome::Error
    } else {
        PresentationOutcome::Success
    };
    let marker = if output.is_error { "✗" } else { "●" };
    let mut headline = format!(
        "{marker} {} · {}",
        action_label(name, arguments, true),
        format_duration(duration)
    );
    if let Some(detail) = outcome_detail(output) {
        headline.push_str(" · ");
        headline.push_str(&detail);
    }

    let mut rendered_output = if let Some(diff) = output
        .details
        .as_ref()
        .and_then(|details| details.get("diff"))
        .and_then(Value::as_str)
    {
        preview_first_lines(diff, MAX_DIFF_LINES)
    } else if should_show_output(name, output) {
        preview_lines(display_output(name, output), MAX_OUTPUT_LINES)
    } else {
        Vec::new()
    };
    if output.content.truncated {
        rendered_output.push("… tool output truncated".into());
    }
    if output
        .details
        .as_ref()
        .and_then(|details| details.get("diff_truncated"))
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        rendered_output.push("… diff truncated".into());
    }

    ToolPresentation {
        headline,
        output: rendered_output,
        outcome,
    }
}

fn action_label(name: &str, arguments: &Value, completed: bool) -> String {
    let verb = match (name, completed) {
        ("read", false) => "Reading",
        ("read", true) => "Read",
        ("bash", false) => "Running",
        ("bash", true) => "Ran",
        ("edit", false) => "Editing",
        ("edit", true) => "Edited",
        ("write", false) => "Writing",
        ("write", true) => "Wrote",
        (_, false) => "Running",
        (_, true) => "Ran",
    };
    let value = match name {
        "read" | "edit" | "write" => string_argument(arguments, "path"),
        "bash" => string_argument(arguments, "command"),
        _ => None,
    }
    .unwrap_or(name);
    format!("{verb} {}", clipped_label(value))
}

fn string_argument<'a>(arguments: &'a Value, key: &str) -> Option<&'a str> {
    arguments.as_object()?.get(key)?.as_str()
}

fn clipped_label(value: &str) -> String {
    let normalized = value
        .chars()
        .map(|character| {
            if character.is_control() {
                ' '
            } else {
                character
            }
        })
        .collect::<String>();
    let normalized = normalized.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= MAX_LABEL_CHARS {
        return normalized;
    }
    let mut clipped = normalized
        .chars()
        .take(MAX_LABEL_CHARS - 1)
        .collect::<String>();
    clipped.push('…');
    clipped
}

fn format_duration(duration: Duration) -> String {
    if duration.as_millis() < 1_000 {
        format!("{}ms", duration.as_millis())
    } else {
        format!("{:.1}s", duration.as_secs_f64())
    }
}

fn outcome_detail(output: &ToolOutput) -> Option<String> {
    let details = output.details.as_ref()?;
    if let Some(exit_code) = details
        .get("exit_code")
        .and_then(Value::as_i64)
        .filter(|exit_code| *exit_code != 0)
    {
        return Some(format!("exit {exit_code}"));
    }
    if details
        .get("timed_out")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Some("timed out".into());
    }
    if details
        .get("cancelled")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        return Some("cancelled".into());
    }
    None
}

fn should_show_output(name: &str, output: &ToolOutput) -> bool {
    !output.content.text.is_empty() && (output.is_error || name == "bash")
}

fn display_output<'a>(name: &str, output: &'a ToolOutput) -> &'a str {
    let text = output.content.text.as_str();
    if name != "bash" || !output.is_error {
        return text;
    }
    let Some((first, rest)) = text.split_once('\n') else {
        return text;
    };
    if first.starts_with("command exited with ")
        || first.starts_with("command timed out after ")
        || first == "command cancelled"
    {
        rest
    } else {
        text
    }
}

fn preview_lines(value: &str, limit: usize) -> Vec<String> {
    let lines = value.lines().map(sanitize_line).collect::<Vec<_>>();
    if lines.len() <= limit {
        return lines;
    }
    let mut preview = vec![format!("… {} earlier lines", lines.len() - limit)];
    preview.extend_from_slice(&lines[lines.len() - limit..]);
    preview
}

fn preview_first_lines(value: &str, limit: usize) -> Vec<String> {
    let lines = value.lines().collect::<Vec<_>>();
    let mut preview = lines
        .iter()
        .take(limit)
        .map(|line| sanitize_line(line))
        .collect::<Vec<_>>();
    if lines.len() > limit {
        preview.push(format!("… {} more diff lines", lines.len() - limit));
    }
    preview
}

fn sanitize_line(line: &str) -> String {
    line.chars()
        .map(|character| {
            if character.is_control() && character != '\t' {
                '�'
            } else {
                character
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ds_agent_core::BoundedText;
    use serde_json::json;

    #[test]
    fn bash_rows_include_command_duration_and_exit_code() {
        let output = ToolOutput {
            content: BoundedText::new("failure", false),
            is_error: true,
            details: Some(json!({ "exit_code": 7 })),
        };

        let presentation = finished_tool(
            "bash",
            &json!({ "command": "cargo test" }),
            &output,
            Duration::from_millis(340),
        );

        assert_eq!(presentation.headline, "✗ Ran cargo test · 340ms · exit 7");
        assert_eq!(presentation.output, ["failure"]);
        assert_eq!(presentation.outcome, PresentationOutcome::Error);
    }

    #[test]
    fn read_rows_hide_success_content() {
        let output = ToolOutput::success(BoundedText::new("whole file", false));

        let presentation = finished_tool(
            "read",
            &json!({ "path": "src/main.rs" }),
            &output,
            Duration::from_millis(12),
        );

        assert_eq!(presentation.headline, "● Read src/main.rs · 12ms");
        assert!(presentation.output.is_empty());
    }

    #[test]
    fn command_output_keeps_only_the_final_five_lines() {
        let output = ToolOutput::success(BoundedText::new("1\n2\n3\n4\n5\n6\n7", true));

        let presentation = finished_tool(
            "bash",
            &json!({ "command": "seq 7" }),
            &output,
            Duration::from_secs(2),
        );

        assert_eq!(
            presentation.output,
            [
                "… 2 earlier lines",
                "3",
                "4",
                "5",
                "6",
                "7",
                "… tool output truncated"
            ]
        );
    }

    #[test]
    fn command_metadata_removes_the_duplicate_failure_prefix() {
        let output = ToolOutput {
            content: BoundedText::new("command exited with exit status: 7\nactual failure", false),
            is_error: true,
            details: Some(json!({ "exit_code": 7 })),
        };

        let presentation = finished_tool(
            "bash",
            &json!({ "command": "false" }),
            &output,
            Duration::from_millis(10),
        );

        assert_eq!(presentation.output, ["actual failure"]);
    }

    #[test]
    fn diff_preview_keeps_headers_and_first_hunk() {
        let diff = (1..=20)
            .map(|line| format!("line {line}"))
            .collect::<Vec<_>>()
            .join("\n");
        let output = ToolOutput::success(BoundedText::new("edited file", false))
            .with_details(json!({ "diff": diff, "diff_truncated": false }));

        let presentation = finished_tool(
            "edit",
            &json!({ "path": "file" }),
            &output,
            Duration::from_millis(10),
        );

        assert_eq!(
            presentation.output.first().map(String::as_str),
            Some("line 1")
        );
        assert_eq!(
            presentation.output.last().map(String::as_str),
            Some("… 8 more diff lines")
        );
    }
}
