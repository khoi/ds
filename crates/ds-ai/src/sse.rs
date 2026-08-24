#[derive(Default)]
pub(crate) struct Decoder {
    buffer: Vec<u8>,
    data: Vec<String>,
}

impl Decoder {
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    pub(crate) fn next_data(&mut self) -> Result<Option<String>, String> {
        loop {
            let Some(end) = self.buffer.iter().position(|byte| *byte == b'\n') else {
                return Ok(None);
            };
            let mut line = self.buffer.drain(..=end).collect::<Vec<_>>();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = std::str::from_utf8(&line).map_err(|error| error.to_string())?;

            if line.is_empty() {
                if self.data.is_empty() {
                    continue;
                }
                return Ok(Some(self.data.drain(..).collect::<Vec<_>>().join("\n")));
            }
            if line.starts_with(':') {
                continue;
            }
            let (field, value) = line.split_once(':').unwrap_or((line, ""));
            if field == "data" {
                self.data
                    .push(value.strip_prefix(' ').unwrap_or(value).into());
            }
        }
    }
}
