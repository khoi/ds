use serde_json::{Map, Value};

const UNSUPPORTED: [&str; 16] = [
    "$ref",
    "$defs",
    "definitions",
    "allOf",
    "oneOf",
    "patternProperties",
    "dependentSchemas",
    "dependencies",
    "unevaluatedProperties",
    "propertyNames",
    "contains",
    "prefixItems",
    "not",
    "if",
    "then",
    "else",
];

pub(crate) fn object(schema: &Value) -> Value {
    serde_json::json!({
        "type": "object",
        "properties": schema.get("properties").cloned().unwrap_or_else(|| serde_json::json!({})),
        "required": schema.get("required").cloned().unwrap_or_else(|| serde_json::json!([])),
    })
}

pub(crate) fn strict(schema: &Value) -> Result<Value, String> {
    let mut schema = schema.clone();
    normalize(&mut schema)?;
    if schema.get("type").and_then(Value::as_str) != Some("object") {
        return Err("root schema must have type object".into());
    }
    Ok(schema)
}

fn normalize(schema: &mut Value) -> Result<(), String> {
    let Some(object) = schema.as_object_mut() else {
        return Err("boolean schemas are unsupported".into());
    };
    for key in UNSUPPORTED {
        if object.contains_key(key) {
            return Err(format!("{key} schemas are unsupported"));
        }
    }
    normalize_any_of(object)?;
    if let Some(items) = object.get_mut("items") {
        if items.is_array() {
            return Err("tuple schemas are unsupported".into());
        }
        normalize(items)?;
    }
    if object.contains_key("properties")
        && object.get("type").and_then(Value::as_str) != Some("object")
    {
        return Err("properties require type object".into());
    }
    if object.get("type").and_then(Value::as_str) != Some("object") {
        return Ok(());
    }
    if object
        .get("additionalProperties")
        .is_some_and(|value| value != &Value::Bool(false))
    {
        return Err("schema-valued or true additionalProperties is unsupported".into());
    }
    let required = required_names(object)?;
    let mut names = Vec::new();
    if let Some(properties) = object.get("properties") {
        let properties = properties
            .as_object()
            .ok_or_else(|| "object properties must be a schema map".to_string())?;
        if required.iter().any(|name| !properties.contains_key(name)) {
            return Err("required contains an unknown property".into());
        }
        names.extend(properties.keys().cloned());
    } else if !required.is_empty() {
        return Err("required contains an unknown property".into());
    }

    if let Some(properties) = object.get_mut("properties") {
        let properties = properties
            .as_object_mut()
            .ok_or_else(|| "object properties must be a schema map".to_string())?;
        for (name, property) in properties.iter_mut() {
            normalize(property)?;
            if !required.contains(name) && !allows_null(property) {
                *property = serde_json::json!({"anyOf": [property.take(), {"type": "null"}]});
            }
        }
    }
    object.insert(
        "required".into(),
        Value::Array(names.into_iter().map(Value::String).collect()),
    );
    object.insert("additionalProperties".into(), Value::Bool(false));
    Ok(())
}

fn normalize_any_of(object: &mut Map<String, Value>) -> Result<(), String> {
    let Some(any_of) = object.get_mut("anyOf") else {
        return Ok(());
    };
    let variants = any_of
        .as_array_mut()
        .filter(|variants| !variants.is_empty())
        .ok_or_else(|| "anyOf must contain at least one schema".to_string())?;
    for variant in variants {
        if structured(variant) {
            return Err("object and array unions are unsupported".into());
        }
        normalize(variant)?;
    }
    Ok(())
}

fn required_names(object: &Map<String, Value>) -> Result<Vec<String>, String> {
    let Some(required) = object.get("required") else {
        return Ok(Vec::new());
    };
    required
        .as_array()
        .ok_or_else(|| "object required must be a string array".to_string())?
        .iter()
        .map(|name| {
            name.as_str()
                .map(str::to_owned)
                .ok_or_else(|| "object required must be a string array".to_string())
        })
        .collect()
}

fn structured(schema: &Value) -> bool {
    schema.as_object().is_some_and(|schema| {
        schema.contains_key("properties")
            || schema.contains_key("items")
            || schema.get("type").is_some_and(|types| match types {
                Value::String(value) => value == "object" || value == "array",
                Value::Array(values) => values.iter().any(|value| {
                    value
                        .as_str()
                        .is_some_and(|value| value == "object" || value == "array")
                }),
                _ => false,
            })
    })
}

fn allows_null(schema: &Value) -> bool {
    schema.as_object().is_some_and(|schema| {
        schema.get("type").is_some_and(|types| match types {
            Value::String(value) => value == "null",
            Value::Array(values) => values.iter().any(|value| value.as_str() == Some("null")),
            _ => false,
        }) || schema.get("const") == Some(&Value::Null)
            || schema
                .get("enum")
                .and_then(Value::as_array)
                .is_some_and(|values| values.contains(&Value::Null))
            || schema
                .get("anyOf")
                .and_then(Value::as_array)
                .is_some_and(|values| values.iter().any(allows_null))
    })
}
