use crate::{AssistantContent, InputContent, UserContent};

pub trait ContentText {
    fn text_blocks(&self) -> Vec<&str>;
}

pub fn content_text<T>(content: &T) -> String
where
    T: ContentText + ?Sized,
{
    content_text_with_separator(content, "\n")
}

pub fn content_text_with_separator<T>(content: &T, separator: &str) -> String
where
    T: ContentText + ?Sized,
{
    content.text_blocks().join(separator)
}

impl ContentText for str {
    fn text_blocks(&self) -> Vec<&str> {
        vec![self]
    }
}

impl ContentText for String {
    fn text_blocks(&self) -> Vec<&str> {
        vec![self]
    }
}

impl ContentText for [AssistantContent] {
    fn text_blocks(&self) -> Vec<&str> {
        self.iter()
            .filter_map(|block| match block {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect()
    }
}

impl ContentText for Vec<AssistantContent> {
    fn text_blocks(&self) -> Vec<&str> {
        self.as_slice().text_blocks()
    }
}

impl ContentText for [InputContent] {
    fn text_blocks(&self) -> Vec<&str> {
        self.iter()
            .filter_map(|block| match block {
                InputContent::Text(text) => Some(text.text.as_str()),
                InputContent::Image(_) => None,
            })
            .collect()
    }
}

impl ContentText for Vec<InputContent> {
    fn text_blocks(&self) -> Vec<&str> {
        self.as_slice().text_blocks()
    }
}

impl ContentText for UserContent {
    fn text_blocks(&self) -> Vec<&str> {
        match self {
            Self::Text(text) => vec![text],
            Self::Blocks(blocks) => blocks.text_blocks(),
        }
    }
}
