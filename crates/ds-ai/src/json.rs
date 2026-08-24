use serde::de::DeserializeOwned;

pub(crate) fn parse<T: DeserializeOwned>(input: &str) -> Result<T, String> {
    serde_json::from_str(input)
        .or_else(|_| serde_json::from_str(&repair(input)))
        .map_err(|error| error.to_string())
}

pub(crate) fn value(input: &str) -> serde_json::Value {
    parse(input).unwrap_or_else(|_| serde_json::json!({}))
}

pub(crate) fn streaming_value(input: &str) -> serde_json::Value {
    parse(input)
        .or_else(|_| parse(&complete(&repair(input))))
        .unwrap_or_else(|_| serde_json::json!({}))
}

fn complete(input: &str) -> String {
    let mut output = String::with_capacity(input.len() + 8);
    let mut closers = Vec::new();
    let mut in_string = false;
    let mut escaped = false;
    for character in input.chars() {
        output.push(character);
        if in_string {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                in_string = false;
            }
            continue;
        }
        match character {
            '"' => in_string = true,
            '{' => closers.push('}'),
            '[' => closers.push(']'),
            '}' | ']' if closers.last() == Some(&character) => {
                closers.pop();
            }
            _ => {}
        }
    }
    if escaped {
        output.push('\\');
    }
    if in_string {
        output.push('"');
    }
    while output.chars().last().is_some_and(char::is_whitespace) {
        output.pop();
    }
    if output.ends_with(':') {
        output.push_str("null");
    } else if output.ends_with(',') {
        output.pop();
    }
    output.extend(closers.into_iter().rev());
    output
}

fn repair(input: &str) -> String {
    let mut output = String::with_capacity(input.len());
    let mut chars = input.chars().peekable();
    let mut in_string = false;
    while let Some(character) = chars.next() {
        if !in_string {
            output.push(character);
            if character == '"' {
                in_string = true;
            }
            continue;
        }
        match character {
            '"' => {
                output.push(character);
                in_string = false;
            }
            '\\' => {
                let Some(escaped) = chars.next() else {
                    output.push_str("\\\\");
                    continue;
                };
                if escaped == 'u' {
                    let digits = chars.clone().take(4).collect::<String>();
                    if digits.len() == 4 && digits.chars().all(|digit| digit.is_ascii_hexdigit()) {
                        output.push_str("\\u");
                        output.push_str(&digits);
                        for _ in 0..4 {
                            chars.next();
                        }
                    } else {
                        output.push_str("\\\\u");
                    }
                } else if matches!(escaped, '"' | '\\' | '/' | 'b' | 'f' | 'n' | 'r' | 't') {
                    output.push('\\');
                    output.push(escaped);
                } else {
                    output.push_str("\\\\");
                    output.push(escaped);
                }
            }
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            character if character.is_control() => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => output.push(character),
        }
    }
    output
}
