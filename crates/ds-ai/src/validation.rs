use crate::{AssistantToolCall, Tool};
use serde_json::{Number, Value};
use thiserror::Error;

#[derive(Debug, Error)]
pub enum ToolValidationError {
    #[error("Tool \"{0}\" not found")]
    ToolNotFound(String),
    #[error("Invalid schema for tool \"{tool}\": {message}")]
    InvalidSchema { tool: String, message: String },
    #[error("Validation failed for tool \"{tool}\":\n{errors}\n\nReceived arguments:\n{arguments}")]
    InvalidArguments {
        tool: String,
        errors: String,
        arguments: String,
    },
}

pub fn validate_tool_call(
    tools: &[Tool],
    tool_call: &AssistantToolCall,
) -> Result<Value, ToolValidationError> {
    let tool = tools
        .iter()
        .find(|tool| tool.name == tool_call.name)
        .ok_or_else(|| ToolValidationError::ToolNotFound(tool_call.name.clone()))?;
    validate_tool_arguments(tool, tool_call)
}

pub fn validate_tool_arguments(
    tool: &Tool,
    tool_call: &AssistantToolCall,
) -> Result<Value, ToolValidationError> {
    validate_local_references(&tool.parameters, &tool.parameters).map_err(|message| {
        ToolValidationError::InvalidSchema {
            tool: tool.name.clone(),
            message,
        }
    })?;
    let validator = jsonschema::JSONSchema::compile(&tool.parameters).map_err(|error| {
        ToolValidationError::InvalidSchema {
            tool: tool.name.clone(),
            message: error.to_string(),
        }
    })?;
    let mut arguments = tool_call.arguments.clone();
    normalize_optional_nulls(&mut arguments, &tool.parameters, &tool.parameters);
    coerce(&mut arguments, &tool.parameters, &tool.parameters);
    let Err(errors) = validator.validate(&arguments) else {
        return Ok(arguments);
    };
    let errors = errors
        .map(|error| {
            let path = error.instance_path.to_string();
            let path = if path.is_empty() {
                "root".into()
            } else {
                path.trim_start_matches('/').replace('/', ".")
            };
            format!("  - {path}: {error}")
        })
        .collect::<Vec<_>>()
        .join("\n");
    Err(ToolValidationError::InvalidArguments {
        tool: tool_call.name.clone(),
        errors,
        arguments: serde_json::to_string_pretty(&tool_call.arguments)
            .unwrap_or_else(|_| tool_call.arguments.to_string()),
    })
}

fn normalize_optional_nulls(value: &mut Value, schema: &Value, root: &Value) {
    let schema = resolve_schema(root, schema);
    match value {
        Value::Array(values) => match schema.get("items") {
            Some(Value::Array(items)) => {
                for (value, schema) in values.iter_mut().zip(items) {
                    normalize_optional_nulls(value, schema, root);
                }
            }
            Some(schema) => {
                for value in values {
                    normalize_optional_nulls(value, schema, root);
                }
            }
            None => {}
        },
        Value::Object(values) => {
            let Some(properties) = schema.get("properties").and_then(Value::as_object) else {
                return;
            };
            let required = schema
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect::<Vec<_>>();
            let names = properties.keys().cloned().collect::<Vec<_>>();
            for name in names {
                let Some(property_schema) = properties.get(&name) else {
                    continue;
                };
                let remove = values.get(&name).is_some_and(Value::is_null)
                    && !required.contains(&name.as_str())
                    && property_schema.get("$ref").is_none()
                    && !valid(root, property_schema, &Value::Null);
                if remove {
                    values.remove(&name);
                } else if let Some(value) = values.get_mut(&name) {
                    normalize_optional_nulls(value, property_schema, root);
                }
            }
        }
        _ => {}
    }
}

fn coerce(value: &mut Value, schema: &Value, root: &Value) {
    let schema = resolve_schema(root, schema);
    if let Some(schemas) = schema.get("allOf").and_then(Value::as_array) {
        for schema in schemas {
            coerce(value, schema, root);
        }
    }
    for keyword in ["anyOf", "oneOf"] {
        if let Some(schemas) = schema.get(keyword).and_then(Value::as_array) {
            coerce_union(value, schemas, root);
        }
    }
    let types = schema_types(schema);
    let matches_union = types.len() > 1 && types.iter().any(|kind| matches_type(value, kind));
    if !types.is_empty() && !matches_union {
        for kind in &types {
            if let Some(coerced) = coerce_primitive(value, kind) {
                *value = coerced;
                break;
            }
        }
    }
    if types.contains(&"object") {
        coerce_object(value, schema, root);
    }
    if types.contains(&"array") {
        coerce_array(value, schema, root);
    }
}

fn coerce_union(value: &mut Value, schemas: &[Value], root: &Value) {
    if schemas.iter().any(|schema| valid(root, schema, value)) {
        return;
    }
    for schema in schemas {
        let mut candidate = value.clone();
        coerce(&mut candidate, schema, root);
        if valid(root, schema, &candidate) {
            *value = candidate;
            return;
        }
    }
}

fn coerce_object(value: &mut Value, schema: &Value, root: &Value) {
    let Value::Object(values) = value else {
        return;
    };
    let properties = schema.get("properties").and_then(Value::as_object);
    if let Some(properties) = properties {
        for (name, property_schema) in properties {
            if let Some(value) = values.get_mut(name) {
                coerce(value, property_schema, root);
            }
        }
    }
    let Some(additional) = schema.get("additionalProperties") else {
        return;
    };
    if !additional.is_object() {
        return;
    }
    let defined = properties.map_or_else(Vec::new, |properties| {
        properties.keys().map(String::as_str).collect()
    });
    for (name, value) in values {
        if !defined.contains(&name.as_str()) {
            coerce(value, additional, root);
        }
    }
}

fn coerce_array(value: &mut Value, schema: &Value, root: &Value) {
    let Value::Array(values) = value else {
        return;
    };
    match schema.get("items") {
        Some(Value::Array(items)) => {
            for (value, schema) in values.iter_mut().zip(items) {
                coerce(value, schema, root);
            }
        }
        Some(schema) if schema.is_object() => {
            for value in values {
                coerce(value, schema, root);
            }
        }
        _ => {}
    }
}

fn schema_types(schema: &Value) -> Vec<&str> {
    match schema.get("type") {
        Some(Value::String(kind)) => vec![kind],
        Some(Value::Array(types)) => types.iter().filter_map(Value::as_str).collect(),
        _ => Vec::new(),
    }
}

fn matches_type(value: &Value, kind: &str) -> bool {
    match kind {
        "number" => value.is_number(),
        "integer" => value.as_f64().is_some_and(|value| value.fract() == 0.0),
        "boolean" => value.is_boolean(),
        "string" => value.is_string(),
        "null" => value.is_null(),
        "array" => value.is_array(),
        "object" => value.is_object(),
        _ => false,
    }
}

fn coerce_primitive(value: &Value, kind: &str) -> Option<Value> {
    match kind {
        "number" => match value {
            Value::Null => Some(Value::Number(Number::from(0))),
            Value::String(value) if !value.trim().is_empty() => value
                .parse::<i64>()
                .ok()
                .map(Number::from)
                .or_else(|| value.parse::<u64>().ok().map(Number::from))
                .or_else(|| {
                    value
                        .parse::<f64>()
                        .ok()
                        .filter(|value| value.is_finite())
                        .and_then(Number::from_f64)
                })
                .map(Value::Number),
            Value::Bool(value) => Some(Value::Number(Number::from(u8::from(*value)))),
            _ => None,
        },
        "integer" => match value {
            Value::Null => Some(Value::Number(Number::from(0))),
            Value::String(value) if !value.trim().is_empty() => value
                .parse::<i64>()
                .ok()
                .map(Number::from)
                .map(Value::Number),
            Value::Bool(value) => Some(Value::Number(Number::from(u8::from(*value)))),
            _ => None,
        },
        "boolean" => match value {
            Value::Null => Some(Value::Bool(false)),
            Value::String(value) if value == "true" => Some(Value::Bool(true)),
            Value::String(value) if value == "false" => Some(Value::Bool(false)),
            Value::Number(value) if value.as_i64() == Some(1) => Some(Value::Bool(true)),
            Value::Number(value) if value.as_i64() == Some(0) => Some(Value::Bool(false)),
            _ => None,
        },
        "string" => match value {
            Value::Null => Some(Value::String(String::new())),
            Value::Bool(value) => Some(Value::String(value.to_string())),
            Value::Number(value) => Some(Value::String(value.to_string())),
            _ => None,
        },
        "null" if value == "" || value == 0 || value == false => Some(Value::Null),
        _ => None,
    }
}

fn valid(root: &Value, schema: &Value, value: &Value) -> bool {
    let schema = resolve_schema(root, schema);
    jsonschema::JSONSchema::compile(schema).is_ok_and(|validator| validator.is_valid(value))
}

fn resolve_schema<'a>(root: &'a Value, schema: &'a Value) -> &'a Value {
    schema
        .get("$ref")
        .and_then(Value::as_str)
        .and_then(|reference| resolve_reference(root, reference))
        .unwrap_or(schema)
}

fn resolve_reference<'a>(root: &'a Value, reference: &str) -> Option<&'a Value> {
    let pointer = reference.strip_prefix('#')?;
    root.pointer(pointer)
}

fn validate_local_references(root: &Value, schema: &Value) -> Result<(), String> {
    match schema {
        Value::Object(schema) => {
            if let Some(reference) = schema.get("$ref") {
                let reference = reference
                    .as_str()
                    .ok_or_else(|| "$ref must be a string".to_string())?;
                if reference.starts_with('#') && resolve_reference(root, reference).is_none() {
                    return Err(format!("unresolved schema reference {reference}"));
                }
            }
            for (name, schema) in schema {
                if !matches!(name.as_str(), "const" | "enum" | "default" | "examples") {
                    validate_local_references(root, schema)?;
                }
            }
            Ok(())
        }
        Value::Array(schemas) => {
            for schema in schemas {
                validate_local_references(root, schema)?;
            }
            Ok(())
        }
        _ => Ok(()),
    }
}
