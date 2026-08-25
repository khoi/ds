use crate::{ConstrainedSampling, ConstrainedSamplingStrictness, Tool, schema};
use serde_json::Value;

pub(crate) struct JsonSchemaSampling {
    pub parameters: Value,
    pub strict: bool,
}

pub(crate) struct GrammarSampling {
    pub syntax: &'static str,
    pub definition: String,
    pub input_property: String,
}

pub(crate) fn json_schema(
    tool: &Tool,
    supported: bool,
) -> Result<Option<JsonSchemaSampling>, String> {
    let Some(ConstrainedSampling::JsonSchema { strict }) = tool.constrained_sampling else {
        return Ok(None);
    };
    if !supported {
        return match strict {
            ConstrainedSamplingStrictness::Prefer => Ok(None),
            ConstrainedSamplingStrictness::Require => Err(format!(
                "Tool {:?} requires JSON-schema constrained sampling, but strict tools are unsupported.",
                tool.name
            )),
        };
    }
    match schema::strict(&tool.parameters) {
        Ok(parameters) => Ok(Some(JsonSchemaSampling {
            parameters,
            strict: true,
        })),
        Err(_) if strict == ConstrainedSamplingStrictness::Prefer => Ok(None),
        Err(error) => Err(format!(
            "Tool {:?} requires JSON-schema constrained sampling, but {error}.",
            tool.name
        )),
    }
}

pub(crate) fn grammar(tool: &Tool, supported: bool) -> Result<Option<GrammarSampling>, String> {
    let Some(ConstrainedSampling::Grammar { variants }) = &tool.constrained_sampling else {
        return Ok(None);
    };
    if !supported {
        return Ok(None);
    }
    let (syntax, definition) = variants
        .openai_lark
        .as_deref()
        .filter(|value| !value.trim().is_empty())
        .map(|value| ("lark", value))
        .or_else(|| {
            variants
                .openai_regex
                .as_deref()
                .filter(|value| !value.trim().is_empty())
                .map(|value| ("regex", value))
        })
        .ok_or_else(|| {
            format!(
                "tool {:?} cannot use grammar constrained sampling: no supported grammar variant was provided",
                tool.name
            )
        })?;
    Ok(Some(GrammarSampling {
        syntax,
        definition: definition.into(),
        input_property: grammar_input_property(tool)?,
    }))
}

pub(crate) fn grammar_input(
    tool_name: &str,
    arguments: &Value,
    input_property: &str,
) -> Result<String, String> {
    arguments
        .get(input_property)
        .and_then(Value::as_str)
        .map(str::to_owned)
        .ok_or_else(|| {
            format!(
                "grammar tool call {tool_name:?} requires argument {input_property:?} to be a string"
            )
        })
}

pub(crate) struct GrammarInputBuffer {
    pub input: String,
    pub started: bool,
    pub closed: bool,
}

pub(crate) fn append_grammar_input_delta(
    buffer: &mut GrammarInputBuffer,
    input_property: &str,
    next_input: &str,
    close: bool,
) -> Result<Option<String>, String> {
    if buffer.closed {
        if close && next_input == buffer.input {
            return Ok(None);
        }
        return Err(format!(
            "grammar tool input for property {input_property:?} changed after it was closed"
        ));
    }
    let Some(input_delta) = next_input.strip_prefix(&buffer.input) else {
        return Err(format!(
            "grammar tool input for property {input_property:?} changed non-monotonically"
        ));
    };
    if !close && input_delta.is_empty() {
        return Ok(None);
    }
    let mut delta = String::new();
    if !buffer.started {
        delta.push('{');
        delta.push_str(&serde_json::to_string(input_property).expect("property serializes"));
        delta.push_str(":\"");
        buffer.started = true;
    }
    let encoded = serde_json::to_string(input_delta).expect("input serializes");
    delta.push_str(&encoded[1..encoded.len() - 1]);
    buffer.input = next_input.into();
    if close {
        delta.push_str("\"}");
        buffer.closed = true;
    }
    Ok(Some(delta))
}

fn grammar_input_property(tool: &Tool) -> Result<String, String> {
    let schema = tool
        .parameters
        .as_object()
        .filter(|schema| schema.get("type").and_then(Value::as_str) == Some("object"))
        .ok_or_else(|| {
            "grammar constrained sampling requires an object parameter schema".to_string()
        })?;
    let required = schema
        .get("required")
        .and_then(Value::as_array)
        .filter(|required| required.len() == 1)
        .and_then(|required| required[0].as_str())
        .ok_or_else(|| {
            "grammar constrained sampling requires exactly one required string property".to_string()
        })?;
    let property = schema
        .get("properties")
        .and_then(Value::as_object)
        .and_then(|properties| properties.get(required))
        .ok_or_else(|| {
            format!("grammar constrained sampling requires a properties entry for {required}")
        })?;
    if property.get("type").and_then(Value::as_str) != Some("string") {
        return Err(format!(
            "grammar constrained sampling property {required} must have type string"
        ));
    }
    Ok(required.into())
}
