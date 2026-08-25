use langchart_adapters::llm::LlmError;

/// Map a `genai::Error` to an [`LlmError`].
pub fn map(err: genai::Error) -> LlmError {
    map_message(&err.to_string())
}

/// Map an error message string to an [`LlmError`].
/// Extracted for testability.
pub(crate) fn map_message(msg: &str) -> LlmError {
    if msg.contains("429") || msg.contains("rate limit") || msg.contains("RateLimit") {
        return LlmError::RateLimited(msg.to_string());
    }
    if msg.contains("context")
        && (msg.contains("length") || msg.contains("window") || msg.contains("exceed"))
    {
        return LlmError::ContextLengthExceeded;
    }
    if msg.contains("timeout") || msg.contains("timed out") {
        return LlmError::Timeout;
    }
    LlmError::Provider(msg.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests operate on the pure string-matching layer (map_message) to avoid
    // the need to construct opaque genai::Error variants in unit tests.

    #[test]
    fn rate_limited_by_429() {
        assert!(matches!(
            map_message("HTTP 429 Too Many Requests"),
            LlmError::RateLimited(_)
        ));
    }

    #[test]
    fn rate_limited_by_phrase() {
        assert!(matches!(
            map_message("rate limit exceeded"),
            LlmError::RateLimited(_)
        ));
    }

    #[test]
    fn rate_limited_case_sensitive_variant() {
        assert!(matches!(
            map_message("RateLimit hit"),
            LlmError::RateLimited(_)
        ));
    }

    #[test]
    fn context_length_exceeded() {
        assert!(matches!(
            map_message("context length exceeded for this model"),
            LlmError::ContextLengthExceeded
        ));
    }

    #[test]
    fn context_window_exceeded() {
        assert!(matches!(
            map_message("context window too large"),
            LlmError::ContextLengthExceeded
        ));
    }

    #[test]
    fn context_exceed() {
        assert!(matches!(
            map_message("context exceed limit"),
            LlmError::ContextLengthExceeded
        ));
    }

    #[test]
    fn timeout_by_word() {
        assert!(matches!(map_message("request timeout"), LlmError::Timeout));
    }

    #[test]
    fn timeout_by_phrase() {
        assert!(matches!(
            map_message("request timed out after 30s"),
            LlmError::Timeout
        ));
    }

    #[test]
    fn generic_provider_error() {
        assert!(matches!(
            map_message("something unexpected happened"),
            LlmError::Provider(_)
        ));
    }

    #[test]
    fn empty_message_is_provider() {
        assert!(matches!(map_message(""), LlmError::Provider(_)));
    }
}
