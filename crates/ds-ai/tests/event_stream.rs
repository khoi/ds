use ds_ai::{
    Api, AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream,
    AssistantMessageStreamError, DoneReason, ErrorReason, ProviderId, StopReason,
};
use futures_util::{StreamExt, stream};

fn message(stop_reason: StopReason) -> AssistantMessage {
    AssistantMessage {
        content: Vec::new(),
        api: Api::OpenAiResponses,
        provider: ProviderId::new("openai"),
        model: "test".into(),
        response_model: None,
        response_id: None,
        diagnostics: None,
        usage: Default::default(),
        stop_reason,
        error_message: None,
        raw_stop_reason: None,
        end_turn: None,
        timestamp: 1,
    }
}

#[tokio::test]
async fn settles_done_and_stops_after_the_terminal_event() {
    let pending = message(StopReason::Pending);
    let done = message(StopReason::Stop);
    let events = [
        AssistantMessageEvent::Start { partial: pending },
        AssistantMessageEvent::Done {
            reason: DoneReason::Stop,
            message: done.clone(),
        },
        AssistantMessageEvent::Start {
            partial: message(StopReason::Pending),
        },
    ];
    let mut events = AssistantMessageEventStream::new(stream::iter(events));

    assert!(matches!(
        events.next().await,
        Some(AssistantMessageEvent::Start { .. })
    ));
    assert!(matches!(
        events.next().await,
        Some(AssistantMessageEvent::Done { .. })
    ));
    assert_eq!(events.next().await, None);
    assert_eq!(events.result().await.unwrap(), done);
}

#[tokio::test]
async fn settles_error_as_the_stream_result() {
    let mut failed = message(StopReason::Error);
    failed.error_message = Some("failed".into());
    let mut events =
        AssistantMessageEventStream::new(stream::iter([AssistantMessageEvent::Error {
            reason: ErrorReason::Error,
            error: failed.clone(),
        }]));

    assert_eq!(events.result().await.unwrap(), failed);
    assert_eq!(events.next().await, None);
}

#[tokio::test]
async fn rejects_a_stream_without_a_terminal_event() {
    let mut events =
        AssistantMessageEventStream::new(stream::iter([AssistantMessageEvent::Start {
            partial: message(StopReason::Pending),
        }]));

    assert_eq!(
        events.result().await,
        Err(AssistantMessageStreamError::MissingTerminalEvent)
    );
}

#[test]
fn terminal_event_reasons_accept_only_their_valid_stop_reasons() {
    assert_eq!(
        [
            StopReason::Stop,
            StopReason::Length,
            StopReason::ToolUse,
            StopReason::Deferred,
        ]
        .map(DoneReason::try_from),
        [
            Ok(DoneReason::Stop),
            Ok(DoneReason::Length),
            Ok(DoneReason::ToolUse),
            Ok(DoneReason::Deferred),
        ]
    );
    assert_eq!(
        [StopReason::Error, StopReason::Aborted].map(ErrorReason::try_from),
        [Ok(ErrorReason::Error), Ok(ErrorReason::Aborted)]
    );
    assert_eq!(
        DoneReason::try_from(StopReason::Error),
        Err(StopReason::Error)
    );
    assert_eq!(
        ErrorReason::try_from(StopReason::Stop),
        Err(StopReason::Stop)
    );
}
