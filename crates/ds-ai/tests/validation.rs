use ds_ai::{
    AssistantToolCall, Tool, ToolValidationError, validate_tool_arguments, validate_tool_call,
};
use serde_json::{Value, json};

#[test]
fn coerces_plain_schema_primitives() {
    let cases = [
        (json!({"type": "number"}), json!("42"), json!(42)),
        (json!({"type": "number"}), json!(true), json!(1)),
        (json!({"type": "number"}), Value::Null, json!(0)),
        (json!({"type": "integer"}), json!("42"), json!(42)),
        (json!({"type": "boolean"}), json!("true"), json!(true)),
        (json!({"type": "boolean"}), json!("false"), json!(false)),
        (json!({"type": "boolean"}), json!(1), json!(true)),
        (json!({"type": "boolean"}), json!(0), json!(false)),
        (json!({"type": "string"}), Value::Null, json!("")),
        (json!({"type": "string"}), json!(true), json!("true")),
        (json!({"type": "null"}), json!(""), Value::Null),
        (json!({"type": "null"}), json!(0), Value::Null),
        (json!({"type": "null"}), json!(false), Value::Null),
        (
            json!({"type": ["number", "string"]}),
            json!("1"),
            json!("1"),
        ),
        (json!({"type": ["boolean", "number"]}), json!("1"), json!(1)),
    ];
    for (schema, input, expected) in cases {
        let (tool, call) = call(schema, input);
        assert_eq!(
            validate_tool_arguments(&tool, &call).unwrap(),
            json!({"value": expected})
        );
    }
}

#[test]
fn removes_optional_non_nullable_nulls_and_keeps_nullable_nulls() {
    let tool = Tool::new(
        "echo",
        "Echo tool",
        json!({
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "offset": {"type": "number"},
                "nullable": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                "metadata": {
                    "type": "object",
                    "properties": {"enabled": {"type": "boolean"}}
                }
            },
            "required": ["path", "metadata"]
        }),
    );
    let call = tool_call(json!({
        "path": "file.txt",
        "offset": null,
        "nullable": null,
        "metadata": {"enabled": null}
    }));
    assert_eq!(
        validate_tool_arguments(&tool, &call).unwrap(),
        json!({"path": "file.txt", "nullable": null, "metadata": {}})
    );
}

#[test]
fn preserves_optional_references_and_matching_nullable_unions() {
    let tool = Tool::new(
        "echo",
        "Echo tool",
        json!({
            "type": "object",
            "properties": {"value": {"$ref": "#/$defs/value"}},
            "$defs": {"value": {"anyOf": [{"type": "number"}, {"type": "null"}]}}
        }),
    );
    assert_eq!(
        validate_tool_arguments(&tool, &tool_call(json!({"value": null}))).unwrap(),
        json!({"value": null})
    );

    for schema in [
        json!({"anyOf": [{"type": "number"}, {"type": "null"}]}),
        json!({"oneOf": [{"type": "number"}, {"type": "null"}]}),
        json!({"type": ["array", "null"], "items": {"type": "string"}}),
    ] {
        let (tool, call) = call(schema, Value::Null);
        assert_eq!(
            validate_tool_arguments(&tool, &call).unwrap(),
            json!({"value": null})
        );
    }
}

#[test]
fn coerces_nested_unions_arrays_and_additional_properties() {
    let tool = Tool::new(
        "echo",
        "Echo tool",
        json!({
            "type": "object",
            "properties": {
                "value": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                "items": {"type": "array", "items": {"type": "boolean"}},
                "metadata": {
                    "type": "object",
                    "additionalProperties": {"type": "integer"}
                }
            },
            "required": ["value", "items", "metadata"]
        }),
    );
    assert_eq!(
        validate_tool_arguments(
            &tool,
            &tool_call(json!({
                "value": "42",
                "items": ["true", 0],
                "metadata": {"first": "1", "second": false}
            }))
        )
        .unwrap(),
        json!({
            "value": 42,
            "items": [true, false],
            "metadata": {"first": 1, "second": 0}
        })
    );
}

#[test]
fn rejects_invalid_arguments_and_missing_tools() {
    for (schema, input) in [
        (json!({"type": "boolean"}), json!("1")),
        (json!({"type": "boolean"}), json!("0")),
        (json!({"type": "null"}), json!("null")),
        (json!({"type": "integer"}), json!("42.1")),
    ] {
        let (tool, call) = call(schema, input);
        assert!(matches!(
            validate_tool_arguments(&tool, &call),
            Err(ToolValidationError::InvalidArguments { .. })
        ));
    }

    let call = tool_call(json!({}));
    assert!(matches!(
        validate_tool_call(&[], &call),
        Err(ToolValidationError::ToolNotFound(name)) if name == "echo"
    ));
}

#[test]
fn validates_common_json_schema_constraints() {
    let tool = Tool::new(
        "echo",
        "Echo tool",
        json!({
            "type": "object",
            "properties": {
                "name": {"type": "string", "minLength": 2, "pattern": "^[a-z]+$"},
                "count": {"type": "integer", "minimum": 1, "maximum": 3},
                "tags": {
                    "type": "array",
                    "minItems": 1,
                    "uniqueItems": true,
                    "items": {"enum": ["one", "two"]}
                }
            },
            "required": ["name", "count", "tags"],
            "additionalProperties": false
        }),
    );
    assert_eq!(
        validate_tool_arguments(
            &tool,
            &tool_call(json!({"name": "valid", "count": 2, "tags": ["one"]}))
        )
        .unwrap(),
        json!({"name": "valid", "count": 2, "tags": ["one"]})
    );
    for arguments in [
        json!({"name": "X", "count": 2, "tags": ["one"]}),
        json!({"name": "valid", "count": 4, "tags": ["one"]}),
        json!({"name": "valid", "count": 2, "tags": ["one", "one"]}),
        json!({"name": "valid", "count": 2, "tags": ["three"]}),
        json!({"name": "valid", "count": 2, "tags": ["one"], "extra": true}),
    ] {
        assert!(matches!(
            validate_tool_arguments(&tool, &tool_call(arguments)),
            Err(ToolValidationError::InvalidArguments { .. })
        ));
    }
}

#[test]
fn rejects_invalid_schemas() {
    for schema in [
        json!({"type": "unknown"}),
        json!({"type": "string", "pattern": "["}),
        json!({"$ref": "#/$defs/missing"}),
    ] {
        let tool = Tool::new("echo", "Echo tool", schema.clone());
        let result = validate_tool_arguments(&tool, &tool_call(json!({})));
        assert!(
            matches!(result, Err(ToolValidationError::InvalidSchema { .. })),
            "{schema}: {result:?}"
        );
    }
}

fn call(schema: Value, value: Value) -> (Tool, AssistantToolCall) {
    (
        Tool::new(
            "echo",
            "Echo tool",
            json!({
                "type": "object",
                "properties": {"value": schema},
                "required": ["value"]
            }),
        ),
        tool_call(json!({"value": value})),
    )
}

fn tool_call(arguments: Value) -> AssistantToolCall {
    AssistantToolCall {
        id: "tool-1".into(),
        name: "echo".into(),
        arguments,
        thought_signature: None,
        namespace: None,
    }
}
