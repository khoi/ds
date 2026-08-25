#[derive(Default)]
pub(crate) struct Decoder {
    buffer: Vec<u8>,
    data: Vec<String>,
    event: Option<String>,
    raw: Vec<String>,
    skip_lf: bool,
}

pub(crate) struct Event {
    pub(crate) event: Option<String>,
    pub(crate) data: String,
    pub(crate) raw: Vec<String>,
}

impl Decoder {
    pub(crate) fn push(&mut self, bytes: &[u8]) {
        self.buffer.extend_from_slice(bytes);
    }

    pub(crate) fn next_event(&mut self) -> Result<Option<Event>, String> {
        loop {
            if self.skip_lf {
                let Some(first) = self.buffer.first() else {
                    return Ok(None);
                };
                if *first == b'\n' {
                    self.buffer.remove(0);
                }
                self.skip_lf = false;
            }
            let Some(end) = self
                .buffer
                .iter()
                .position(|byte| matches!(*byte, b'\r' | b'\n'))
            else {
                return Ok(None);
            };
            let mut line = self.buffer.drain(..=end).collect::<Vec<_>>();
            if line.pop() == Some(b'\r') {
                self.skip_lf = true;
            }
            let line = std::str::from_utf8(&line).map_err(|error| error.to_string())?;

            if line.is_empty() {
                if self.data.is_empty() {
                    self.event = None;
                    self.raw.clear();
                    continue;
                }
                return Ok(Some(Event {
                    event: self.event.take(),
                    data: self.data.drain(..).collect::<Vec<_>>().join("\n"),
                    raw: self.raw.drain(..).collect(),
                }));
            }
            self.raw.push(line.into());
            if line.starts_with(':') {
                continue;
            }
            let (field, value) = line.split_once(':').unwrap_or((line, ""));
            match field {
                "event" => self.event = Some(value.strip_prefix(' ').unwrap_or(value).into()),
                "data" => self
                    .data
                    .push(value.strip_prefix(' ').unwrap_or(value).into()),
                _ => {}
            }
        }
    }

    pub(crate) fn finish_event(&mut self) -> Result<Option<Event>, String> {
        if !self.buffer.is_empty() {
            self.buffer.push(b'\n');
            if let Some(event) = self.next_event()? {
                return Ok(Some(event));
            }
        }
        if self.event.is_none() && self.data.is_empty() {
            return Ok(None);
        }
        Ok(Some(Event {
            event: self.event.take(),
            data: self.data.drain(..).collect::<Vec<_>>().join("\n"),
            raw: self.raw.drain(..).collect(),
        }))
    }
}
