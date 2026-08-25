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
    if let Ok(value) = parse(input) {
        return value;
    }
    if let Ok(value) = partial_value(input) {
        return value;
    }
    let repaired = repair(input);
    partial_value(&repaired).unwrap_or_else(|_| serde_json::json!({}))
}

fn partial_value(input: &str) -> Result<serde_json::Value, ()> {
    let input = input.trim();
    PartialParser {
        input: input.as_bytes(),
        index: 0,
    }
    .value()
}

struct PartialParser<'a> {
    input: &'a [u8],
    index: usize,
}

impl PartialParser<'_> {
    fn value(&mut self) -> Result<serde_json::Value, ()> {
        self.skip_whitespace();
        match self.current().ok_or(())? {
            b'"' => self.string().map(serde_json::Value::String),
            b'{' => self.object(),
            b'[' => self.array(),
            _ if self.literal("null") => Ok(serde_json::Value::Null),
            _ if self.literal("true") => Ok(serde_json::Value::Bool(true)),
            _ if self.literal("false") => Ok(serde_json::Value::Bool(false)),
            _ => self.number(),
        }
    }

    fn object(&mut self) -> Result<serde_json::Value, ()> {
        self.index += 1;
        let mut object = serde_json::Map::new();
        loop {
            self.skip_whitespace();
            match self.current() {
                Some(b'}') => {
                    self.index += 1;
                    return Ok(serde_json::Value::Object(object));
                }
                None => return Ok(serde_json::Value::Object(object)),
                _ => {}
            }
            let Ok(key) = self.string() else {
                return Ok(serde_json::Value::Object(object));
            };
            self.skip_whitespace();
            if self.current() != Some(b':') {
                return Ok(serde_json::Value::Object(object));
            }
            self.index += 1;
            let Ok(value) = self.value() else {
                return Ok(serde_json::Value::Object(object));
            };
            object.insert(key, value);
            self.skip_whitespace();
            if self.current() == Some(b',') {
                self.index += 1;
            }
        }
    }

    fn array(&mut self) -> Result<serde_json::Value, ()> {
        self.index += 1;
        let mut array = Vec::new();
        loop {
            self.skip_whitespace();
            match self.current() {
                Some(b']') => {
                    self.index += 1;
                    return Ok(serde_json::Value::Array(array));
                }
                None => return Ok(serde_json::Value::Array(array)),
                _ => {}
            }
            let Ok(value) = self.value() else {
                return Ok(serde_json::Value::Array(array));
            };
            array.push(value);
            self.skip_whitespace();
            if self.current() == Some(b',') {
                self.index += 1;
            }
        }
    }

    fn string(&mut self) -> Result<String, ()> {
        if self.current() != Some(b'"') {
            return Err(());
        }
        let start = self.index;
        self.index += 1;
        let mut escaped = false;
        while let Some(character) = self.current() {
            if character == b'"' && !escaped {
                self.index += 1;
                return serde_json::from_slice(&self.input[start..self.index]).map_err(|_| ());
            }
            escaped = character == b'\\' && !escaped;
            if character != b'\\' {
                escaped = false;
            }
            self.index += 1;
        }
        let end = self.index.saturating_sub(usize::from(escaped));
        let mut candidate = self.input[start..end].to_vec();
        candidate.push(b'"');
        serde_json::from_slice(&candidate).or_else(|_| {
            let slash = self.input[start..end]
                .iter()
                .rposition(|character| *character == b'\\')
                .ok_or(())?;
            let mut candidate = self.input[start..start + slash].to_vec();
            candidate.push(b'"');
            serde_json::from_slice(&candidate).map_err(|_| ())
        })
    }

    fn number(&mut self) -> Result<serde_json::Value, ()> {
        let start = self.index;
        while self
            .current()
            .is_some_and(|character| !matches!(character, b',' | b']' | b'}'))
        {
            self.index += 1;
        }
        let token = std::str::from_utf8(&self.input[start..self.index]).map_err(|_| ())?;
        serde_json::from_str(token).or_else(|_| {
            let exponent = token.rfind('e').ok_or(())?;
            serde_json::from_str(&token[..exponent]).map_err(|_| ())
        })
    }

    fn literal(&mut self, literal: &str) -> bool {
        let remaining = &self.input[self.index..];
        let literal = literal.as_bytes();
        if remaining.starts_with(literal) {
            self.index += literal.len();
            return true;
        }
        if literal.starts_with(remaining) {
            self.index = self.input.len();
            return true;
        }
        false
    }

    fn skip_whitespace(&mut self) {
        while self
            .current()
            .is_some_and(|character| character.is_ascii_whitespace())
        {
            self.index += 1;
        }
    }

    fn current(&self) -> Option<u8> {
        self.input.get(self.index).copied()
    }
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
