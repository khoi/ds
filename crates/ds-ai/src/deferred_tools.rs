use crate::{AssistantContent, Context, Message, Tool};
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum DeferredToolsMode {
    AdditionalTools,
    ToolSearch,
}

pub(crate) struct ToolPlacement {
    pub immediate: Vec<Tool>,
    pub deferred: Vec<(String, Tool)>,
}

impl ToolPlacement {
    pub(crate) fn deferred_tool(&self, name: &str) -> Option<&Tool> {
        self.deferred
            .iter()
            .find_map(|(key, tool)| (key == name).then_some(tool))
    }
}

pub(crate) fn split(
    context: &Context,
    enabled: bool,
    normalize: impl Fn(&str) -> String,
) -> ToolPlacement {
    let mut unique = Vec::<(String, Tool)>::new();
    for tool in context.tools() {
        let name = normalize(&tool.name);
        match unique.iter_mut().find(|(key, _)| key == &name) {
            Some((_, current)) => *current = tool.clone(),
            None => unique.push((name, tool.clone())),
        }
    }
    if !enabled {
        return ToolPlacement {
            immediate: unique.into_iter().map(|(_, tool)| tool).collect(),
            deferred: Vec::new(),
        };
    }

    let mut deferred_names = BTreeSet::new();
    let mut used_names = BTreeSet::new();
    for message in context.messages() {
        match message {
            Message::Assistant(message) => {
                for block in &message.content {
                    if let AssistantContent::ToolCall(call) = block {
                        used_names.insert(normalize(&call.name));
                    }
                }
            }
            Message::ToolResult(result) => {
                for name in result.added_tool_names.iter().flatten() {
                    let name = normalize(name);
                    if !used_names.contains(&name) {
                        deferred_names.insert(name);
                    }
                }
            }
            Message::User(_) => {}
        }
    }

    let mut immediate = Vec::new();
    let mut deferred = Vec::new();
    for (name, tool) in unique {
        if deferred_names.contains(&name) {
            deferred.push((name, tool));
        } else {
            immediate.push(tool);
        }
    }
    ToolPlacement {
        immediate,
        deferred,
    }
}
