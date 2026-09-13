// Copyright 2026 Hans W. Uhlig
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! AWS Bedrock LLM Adapter for Langchart
//!
//! Provides an asynchronous [`LlmAdapter`] implementation that routes chat and completion
//! requests to AWS Bedrock foundation models using the uniform Bedrock Converse API.

use async_trait::async_trait;
#[cfg(feature = "control-plane")]
use aws_sdk_bedrock::Client as BedrockControlPlaneClient;
use aws_sdk_bedrockruntime::{
    Client as BedrockClient,
    error::SdkError,
    operation::converse::ConverseError,
    types::{
        ContentBlock, ConversationRole, ConverseOutput, InferenceConfiguration,
        Message as BedrockMessage, StopReason, SystemContentBlock,
    },
};
use aws_smithy_runtime_api::box_error::BoxError;
use aws_smithy_runtime_api::client::interceptors::Intercept;
use aws_smithy_runtime_api::client::interceptors::context::BeforeTransmitInterceptorContextMut;
use aws_smithy_runtime_api::client::orchestrator::HttpResponse;
use aws_smithy_runtime_api::client::runtime_components::RuntimeComponents;
use aws_smithy_types::config_bag::ConfigBag;
use langchart_adapters::llm::{
    FinishReason, LlmAdapter, LlmError, LlmRequest, LlmResponse, Message, ModelInfo,
    ResponseFormat, TokenUsage, TransportStage,
};
use std::collections::HashSet;
use tokio::sync::OnceCell;

#[derive(Debug)]
struct BearerTokenInterceptor {
    token: String,
}

impl Intercept for BearerTokenInterceptor {
    fn name(&self) -> &'static str {
        "BedrockBearerTokenInterceptor"
    }

    fn modify_before_transmit(
        &self,
        context: &mut BeforeTransmitInterceptorContextMut<'_>,
        _runtime_components: &RuntimeComponents,
        _cfg: &mut ConfigBag,
    ) -> Result<(), BoxError> {
        let headers = context.request_mut().headers_mut();
        headers.insert("authorization", format!("Bearer {}", self.token.trim()));
        Ok(())
    }
}

/// Configuration for the Bedrock client and endpoint.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BedrockConfig {
    /// AWS region (e.g. `us-east-1`, `us-west-2`).
    pub region: String,
    /// Optional endpoint URL override (useful for VPC endpoints or local mocking).
    pub endpoint_url: Option<String>,
    /// Optional AWS named profile in `~/.aws/credentials`.
    pub profile_name: Option<String>,
    /// Optional default AWS Bedrock inference profile ID or ARN (e.g. `us.anthropic.claude-3-7-sonnet-20250219-v1:0`).
    ///
    /// Note: An inference profile is a model identifier and can also be passed
    /// directly as `request.model_policy.model` or registered via [`Self::with_model`].
    pub inference_profile: Option<String>,
    /// Optional list of recognized model IDs or inference profile IDs.
    pub models: Vec<String>,
}

impl BedrockConfig {
    /// Creates a new configuration with a target AWS region.
    #[must_use]
    pub fn new(region: impl Into<String>) -> Self {
        Self {
            region: region.into(),
            endpoint_url: None,
            profile_name: None,
            inference_profile: None,
            models: Vec::new(),
        }
    }

    /// Adds a model ID or inference profile ID to the list of recognized models.
    #[must_use]
    pub fn with_model(mut self, model: impl Into<String>) -> Self {
        self.models.push(model.into());
        self
    }

    /// Adds multiple model IDs or inference profile IDs to the list of recognized models.
    #[must_use]
    pub fn with_models<I, S>(mut self, models: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.models.extend(models.into_iter().map(Into::into));
        self
    }

    /// Sets the Bedrock inference profile ID or ARN.
    #[must_use]
    pub fn with_inference_profile(mut self, profile: impl Into<String>) -> Self {
        self.inference_profile = Some(profile.into());
        self
    }

    /// Sets an optional custom endpoint URL.
    #[must_use]
    pub fn with_endpoint_url(mut self, endpoint_url: impl Into<String>) -> Self {
        self.endpoint_url = Some(endpoint_url.into());
        self
    }

    /// Sets an optional AWS named profile.
    #[must_use]
    pub fn with_profile_name(mut self, profile_name: impl Into<String>) -> Self {
        self.profile_name = Some(profile_name.into());
        self
    }
}

/// Credential options for AWS Bedrock.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum BedrockCredentials {
    /// Explicit static AWS access key, secret access key, and optional session token.
    Static {
        access_key_id: String,
        secret_access_key: String,
        session_token: Option<String>,
    },
    /// Bearer token / API key (e.g. `AWS_BEARER_TOKEN_BEDROCK`) for Bedrock endpoint.
    BearerToken(String),
    /// Rely on the default AWS credential provider chain (environment variables, SSO, IAM roles, profiles, or `AWS_BEARER_TOKEN_BEDROCK`).
    EnvironmentOrProfile,
}

/// Determines if the provided model identifier is an AWS Bedrock inference profile ID or ARN.
///
/// Recognized patterns:
/// - System-defined regional inference profiles (`us.*`, `eu.*`, `apac.*`, `cr.*`)
/// - System-defined global inference profiles (`global.*`)
/// - Application inference profile ARNs (`arn:aws:bedrock:...` or general `arn:aws:`)
#[must_use]
pub fn is_inference_profile(model_or_profile: &str) -> bool {
    let trimmed = model_or_profile.trim();
    trimmed.starts_with("us.")
        || trimmed.starts_with("eu.")
        || trimmed.starts_with("apac.")
        || trimmed.starts_with("cr.")
        || trimmed.starts_with("global.")
        || trimmed.starts_with("arn:aws:")
}

/// Bedrock LLM Adapter implementing [`LlmAdapter`].
pub struct BedrockAdapter {
    config: BedrockConfig,
    credentials: BedrockCredentials,
    client: OnceCell<BedrockClient>,
    #[cfg(feature = "control-plane")]
    control_plane_client: OnceCell<Option<BedrockControlPlaneClient>>,
}

impl BedrockAdapter {
    /// Creates a new `BedrockAdapter` instance.
    pub fn new(config: BedrockConfig, credentials: BedrockCredentials) -> Result<Self, LlmError> {
        Ok(Self {
            config,
            credentials,
            client: OnceCell::new(),
            #[cfg(feature = "control-plane")]
            control_plane_client: OnceCell::new(),
        })
    }

    /// Obtains or initializes the underlying AWS Bedrock Runtime client.
    pub async fn client(&self) -> &BedrockClient {
        self.client
            .get_or_init(|| async {
                let mut config_loader = aws_config::defaults(aws_config::BehaviorVersion::latest())
                    .region(aws_config::Region::new(self.config.region.clone()));

                if let Some(endpoint) = &self.config.endpoint_url {
                    config_loader = config_loader.endpoint_url(endpoint);
                }
                if let Some(profile) = &self.config.profile_name {
                    config_loader = config_loader.profile_name(profile);
                }

                let bearer_token = match &self.credentials {
                    BedrockCredentials::BearerToken(token) => Some(token.clone()),
                    BedrockCredentials::EnvironmentOrProfile => {
                        std::env::var("AWS_BEARER_TOKEN_BEDROCK")
                            .ok()
                            .or_else(|| std::env::var("AWS_BEARER_TOKEN").ok())
                            .or_else(|| std::env::var("BEDROCK_API_KEY").ok())
                            .filter(|t| !t.trim().is_empty())
                    }
                    BedrockCredentials::Static {
                        access_key_id,
                        secret_access_key,
                        session_token,
                    } => {
                        let creds = aws_credential_types::Credentials::new(
                            access_key_id.clone(),
                            secret_access_key.clone(),
                            session_token.clone(),
                            None,
                            "langchart-bedrock-static",
                        );
                        config_loader = config_loader.credentials_provider(creds);
                        None
                    }
                };

                if let Some(token) = bearer_token {
                    // Provide dummy credentials so config loading doesn't fail on missing AWS IAM credentials
                    let creds = aws_credential_types::Credentials::new(
                        "bearer",
                        "bearer",
                        None,
                        None,
                        "langchart-bedrock-bearer",
                    );
                    config_loader = config_loader.credentials_provider(creds);
                    let sdk_config = config_loader.load().await;
                    let bedrock_config = aws_sdk_bedrockruntime::config::Builder::from(&sdk_config)
                        .interceptor(BearerTokenInterceptor { token })
                        .build();
                    BedrockClient::from_conf(bedrock_config)
                } else {
                    let sdk_config = config_loader.load().await;
                    BedrockClient::new(&sdk_config)
                }
            })
            .await
    }

    /// Obtains or initializes the underlying AWS Bedrock Control Plane client when available.
    #[cfg(feature = "control-plane")]
    async fn control_plane_client(&self) -> Option<&BedrockControlPlaneClient> {
        self.control_plane_client
            .get_or_init(|| async {
                match &self.credentials {
                    BedrockCredentials::BearerToken(_) => None,
                    BedrockCredentials::Static {
                        access_key_id,
                        secret_access_key,
                        session_token,
                    } => {
                        let creds = aws_credential_types::Credentials::new(
                            access_key_id.clone(),
                            secret_access_key.clone(),
                            session_token.clone(),
                            None,
                            "langchart-bedrock-static",
                        );
                        let mut config_loader =
                            aws_config::defaults(aws_config::BehaviorVersion::latest())
                                .region(aws_config::Region::new(self.config.region.clone()))
                                .credentials_provider(creds);
                        if let Some(endpoint) = &self.config.endpoint_url {
                            config_loader = config_loader.endpoint_url(endpoint);
                        }
                        let sdk_config = config_loader.load().await;
                        Some(BedrockControlPlaneClient::new(&sdk_config))
                    }
                    BedrockCredentials::EnvironmentOrProfile => {
                        let has_bearer = std::env::var("AWS_BEARER_TOKEN_BEDROCK")
                            .ok()
                            .or_else(|| std::env::var("AWS_BEARER_TOKEN").ok())
                            .or_else(|| std::env::var("BEDROCK_API_KEY").ok())
                            .filter(|t| !t.trim().is_empty())
                            .is_some();
                        if has_bearer {
                            return None;
                        }

                        let mut config_loader =
                            aws_config::defaults(aws_config::BehaviorVersion::latest())
                                .region(aws_config::Region::new(self.config.region.clone()));
                        if let Some(endpoint) = &self.config.endpoint_url {
                            config_loader = config_loader.endpoint_url(endpoint);
                        }
                        if let Some(profile) = &self.config.profile_name {
                            config_loader = config_loader.profile_name(profile);
                        }
                        let sdk_config = config_loader.load().await;
                        Some(BedrockControlPlaneClient::new(&sdk_config))
                    }
                }
            })
            .await
            .as_ref()
    }

    /// Fetches models and inference profiles from the Bedrock Control Plane.
    #[cfg(feature = "control-plane")]
    async fn fetch_control_plane_models(
        &self,
        client: &BedrockControlPlaneClient,
    ) -> Result<Vec<ModelInfo>, LlmError> {
        let mut models = Vec::new();

        if let Ok(resp) = client.list_inference_profiles().send().await {
            for summary in resp.inference_profile_summaries() {
                models.push(ModelInfo {
                    id: summary.inference_profile_id().to_owned(),
                    description: summary.description().map(str::to_owned).or_else(|| {
                        Some(format!(
                            "{} (Inference Profile)",
                            summary.inference_profile_name()
                        ))
                    }),
                });
            }
        }

        if let Ok(resp) = client.list_foundation_models().send().await {
            for summary in resp.model_summaries() {
                models.push(ModelInfo {
                    id: summary.model_id().to_owned(),
                    description: summary.model_name().map(str::to_owned),
                });
            }
        }

        Ok(models)
    }

    /// Returns the Bedrock configuration.
    #[must_use]
    pub fn config(&self) -> &BedrockConfig {
        &self.config
    }

    /// Resolves an inference profile ID if available from the request, configuration,
    /// environment, or regional fallback for known foundation models.
    #[must_use]
    pub fn resolve_inference_profile(
        &self,
        request: &LlmRequest,
        primary_model: Option<&str>,
    ) -> Option<String> {
        // 1. Explicit profile in model policy (if not empty or the placeholder "default")
        if let Some(profile) = &request.model_policy.profile {
            let trimmed = profile.trim();
            if !trimmed.is_empty() && trimmed != "default" {
                return Some(trimmed.to_owned());
            }
        }

        // 2. If primary model is already an inference profile or ARN, return it directly
        if let Some(model) = primary_model {
            let trimmed = model.trim();
            if is_inference_profile(trimmed) {
                return Some(trimmed.to_owned());
            }
        }

        // 3. Regional system inference profile fallback for bare foundation models
        if let Some(model) = primary_model {
            let trimmed = model.trim();
            if !is_inference_profile(trimmed) {
                let prefix = if self.config.region.starts_with("us-") {
                    Some("us.")
                } else if self.config.region.starts_with("eu-") {
                    Some("eu.")
                } else if self.config.region.starts_with("ap-") {
                    Some("apac.")
                } else {
                    None
                };

                if let Some(p) = prefix {
                    return Some(format!("{p}{trimmed}"));
                }
            }
        }

        // 4. Explicit inference_profile in BedrockConfig (when no primary model was specified or as fallback)
        if let Some(profile) = &self.config.inference_profile {
            let trimmed = profile.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_owned());
            }
        }

        // 5. Environment variables
        if let Some(profile) = std::env::var("AWS_BEDROCK_INFERENCE_PROFILE_ID")
            .ok()
            .or_else(|| std::env::var("AWS_BEDROCK_INFERENCE_PROFILE").ok())
            .or_else(|| std::env::var("BEDROCK_INFERENCE_PROFILE").ok())
            .filter(|s| !s.trim().is_empty())
        {
            return Some(profile.trim().to_owned());
        }

        None
    }
}

fn format_error_chain(err: &dyn std::error::Error) -> String {
    let mut parts = Vec::new();
    let mut current: Option<&dyn std::error::Error> = Some(err);
    while let Some(e) = current {
        let s = e.to_string();
        if !s.is_empty() && !parts.contains(&s) {
            parts.push(s);
        }
        current = e.source();
    }
    if parts.is_empty() {
        format!("{err:?}")
    } else {
        parts.join(": ")
    }
}

fn format_converse_unhandled(err: &ConverseError) -> String {
    let meta = err.meta();
    let code = meta.code();
    let message = meta.message();

    let mut parts = Vec::new();

    match (code, message) {
        (Some(c), Some(m)) => parts.push(format!("[{c}] {m}")),
        (Some(c), None) => parts.push(format!("[{c}]")),
        (None, Some(m)) => parts.push(m.to_owned()),
        (None, None) => {}
    }

    let mut current = std::error::Error::source(err);
    while let Some(e) = current {
        let s = e.to_string();
        if !s.is_empty() && !parts.iter().any(|p| p.contains(&s)) {
            parts.push(s);
        }
        current = e.source();
    }

    if parts.is_empty() {
        let display = err.to_string();
        if display.is_empty() || display == "unhandled error" {
            format!("Bedrock unhandled error: {err:?}")
        } else {
            format!("Bedrock provider error: {display}")
        }
    } else {
        format!("Bedrock unhandled error: {}", parts.join(": "))
    }
}

fn map_converse_error(err: ConverseError, model_id: &str) -> LlmError {
    match err {
        ConverseError::ThrottlingException(e) => {
            LlmError::RateLimited(e.message().unwrap_or("throttled by Bedrock").to_owned())
        }
        ConverseError::ModelNotReadyException(e) => LlmError::ModelNotFound {
            model: e.message().unwrap_or(model_id).to_owned(),
        },
        ConverseError::ResourceNotFoundException(e) => LlmError::ModelNotFound {
            model: e.message().unwrap_or(model_id).to_owned(),
        },
        ConverseError::AccessDeniedException(e) => {
            let msg = e.message().unwrap_or("access denied to Bedrock model");
            LlmError::Provider(format!("Bedrock access denied for `{model_id}`: {msg}"))
        }
        ConverseError::ValidationException(e) => {
            let msg = e.message().unwrap_or("validation error");
            if msg.contains("context length") || msg.contains("max tokens") {
                LlmError::ContextLengthExceeded
            } else {
                LlmError::Provider(format!("Bedrock validation error: {msg}"))
            }
        }
        ConverseError::ModelTimeoutException(e) => {
            let msg = e.message().unwrap_or("model execution timed out");
            LlmError::Provider(format!("Bedrock model timed out for `{model_id}`: {msg}"))
        }
        ConverseError::InternalServerException(e) => {
            let msg = e.message().unwrap_or("internal server error");
            LlmError::Provider(format!("Bedrock internal server error: {msg}"))
        }
        ConverseError::ServiceUnavailableException(e) => {
            let msg = e.message().unwrap_or("service unavailable");
            LlmError::Provider(format!("Bedrock service unavailable: {msg}"))
        }
        ConverseError::ModelErrorException(e) => {
            let msg = e.message().unwrap_or("model error");
            LlmError::Provider(format!("Bedrock model error for `{model_id}`: {msg}"))
        }
        other => LlmError::Provider(format_converse_unhandled(&other)),
    }
}

fn map_bedrock_sdk_error(err: SdkError<ConverseError, HttpResponse>, model_id: &str) -> LlmError {
    match err {
        SdkError::TimeoutError(_) => {
            let cause = format_error_chain(&err);
            LlmError::Transport {
                stage: TransportStage::Send,
                retryable: true,
                cause: format!("Bedrock request timed out: {cause}"),
            }
        }
        SdkError::DispatchFailure(ref e) => {
            let stage = if e.is_io() {
                TransportStage::Connect
            } else {
                TransportStage::Send
            };
            let cause = format_error_chain(&err);
            LlmError::Transport {
                stage,
                retryable: true,
                cause: format!("Bedrock dispatch failure: {cause}"),
            }
        }
        SdkError::ResponseError(ref e) => {
            let status = e.raw().status().as_u16();
            let cause = format_error_chain(&err);
            if status == 429 {
                LlmError::RateLimited(format!("Bedrock rate limited (HTTP 429): {cause}"))
            } else if status == 408 || status == 504 {
                LlmError::Transport {
                    stage: TransportStage::Headers,
                    retryable: true,
                    cause: format!("Bedrock gateway/request timeout (HTTP {status}): {cause}"),
                }
            } else {
                LlmError::Provider(format!("Bedrock HTTP {status} response error: {cause}"))
            }
        }
        SdkError::ConstructionFailure(_) => {
            let cause = format_error_chain(&err);
            LlmError::Provider(format!("failed to construct Bedrock request: {cause}"))
        }
        SdkError::ServiceError(e) => map_converse_error(e.into_err(), model_id),
        other => {
            let cause = format_error_chain(&other);
            LlmError::Provider(format!("Bedrock SDK error: {cause}"))
        }
    }
}

fn is_model_denied_or_inference_profile_required(err: &ConverseError) -> bool {
    match err {
        ConverseError::AccessDeniedException(_) => true,
        ConverseError::ValidationException(e) => {
            let msg = e.message().unwrap_or("").to_lowercase();
            msg.contains("inference profile")
                || msg.contains("on-demand throughput")
                || msg.contains("not supported")
                || msg.contains("denied")
        }
        ConverseError::ResourceNotFoundException(_) => true,
        other => {
            let meta_code = other.meta().code().unwrap_or("").to_lowercase();
            let meta_msg = other.meta().message().unwrap_or("").to_lowercase();
            let display_msg = other.to_string().to_lowercase();
            meta_code.contains("accessdenied")
                || meta_code.contains("denied")
                || meta_msg.contains("accessdenied")
                || meta_msg.contains("inference profile")
                || meta_msg.contains("on-demand throughput")
                || meta_msg.contains("not supported")
                || meta_msg.contains("denied")
                || display_msg.contains("accessdenied")
                || display_msg.contains("inference profile")
                || display_msg.contains("denied")
        }
    }
}

#[async_trait]
impl LlmAdapter for BedrockAdapter {
    async fn complete(&self, request: LlmRequest) -> Result<LlmResponse, LlmError> {
        if request.response_format != ResponseFormat::Text {
            return Err(LlmError::UnsupportedResponseFormat {
                adapter: "bedrock".to_owned(),
                requested: request.response_format.kind(),
            });
        }
        if !request.tools.is_empty() {
            return Err(LlmError::Provider(
                "bedrock adapter does not yet support tool calls".to_owned(),
            ));
        }

        let primary_model_id = request.model_policy.model.clone();
        let explicit_profile = request
            .model_policy
            .profile
            .as_deref()
            .map(str::trim)
            .filter(|p| !p.is_empty() && *p != "default");

        let target_model_id = if let Some(prof) = explicit_profile {
            prof.to_owned()
        } else if let Some(ref model) = primary_model_id {
            model.clone()
        } else if let Some(ref profile) = self.config.inference_profile {
            profile.clone()
        } else {
            return Err(LlmError::Provider(
                "model ID or inference profile must be specified for Bedrock request".to_owned(),
            ));
        };

        let fallback_profile_id =
            self.resolve_inference_profile(&request, primary_model_id.as_deref());

        let mut system_blocks = Vec::new();
        let mut messages = Vec::new();

        for msg in &request.messages {
            match msg {
                Message::System { content } => {
                    system_blocks.push(SystemContentBlock::Text(content.clone()));
                }
                Message::User { content } => {
                    let bedrock_msg = BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::Text(content.clone()))
                        .build()
                        .map_err(|e| {
                            LlmError::Provider(format!("failed to build user message: {e}"))
                        })?;
                    messages.push(bedrock_msg);
                }
                Message::Assistant { content, .. } => {
                    let bedrock_msg = BedrockMessage::builder()
                        .role(ConversationRole::Assistant)
                        .content(ContentBlock::Text(content.clone()))
                        .build()
                        .map_err(|e| {
                            LlmError::Provider(format!("failed to build assistant message: {e}"))
                        })?;
                    messages.push(bedrock_msg);
                }
                Message::Tool { content, .. } => {
                    let bedrock_msg = BedrockMessage::builder()
                        .role(ConversationRole::User)
                        .content(ContentBlock::Text(content.clone()))
                        .build()
                        .map_err(|e| {
                            LlmError::Provider(format!("failed to build tool message: {e}"))
                        })?;
                    messages.push(bedrock_msg);
                }
            }
        }

        let mut inference_builder = InferenceConfiguration::builder();
        if let Some(tokens) = request
            .model_policy
            .max_tokens
            .and_then(|t| i32::try_from(t).ok())
        {
            inference_builder = inference_builder.max_tokens(tokens);
        }
        if let Some(temp) = request.model_policy.temperature {
            inference_builder = inference_builder.temperature(temp);
        }
        let inference_config = inference_builder.build();

        let client = self.client().await;

        let send_converse = |model_to_use: &str| {
            let mut converse_builder = client
                .converse()
                .model_id(model_to_use)
                .set_messages(Some(messages.clone()))
                .inference_config(inference_config.clone());

            if !system_blocks.is_empty() {
                converse_builder = converse_builder.set_system(Some(system_blocks.clone()));
            }
            converse_builder
        };

        let (response, reported_model) = match send_converse(&target_model_id).send().await {
            Ok(res) => {
                let reported = if primary_model_id.is_none() {
                    Some(target_model_id.clone())
                } else {
                    None
                };
                (res, reported)
            }
            Err(err) => match err {
                SdkError::ServiceError(service_err) => {
                    let converse_err = service_err.into_err();
                    if is_model_denied_or_inference_profile_required(&converse_err) {
                        if let Some(ref fallback_profile) = fallback_profile_id {
                            if fallback_profile != &target_model_id {
                                match send_converse(fallback_profile).send().await {
                                    Ok(res) => (res, Some(fallback_profile.clone())),
                                    Err(retry_err) => {
                                        return Err(map_bedrock_sdk_error(
                                            retry_err,
                                            fallback_profile,
                                        ));
                                    }
                                }
                            } else {
                                return Err(map_converse_error(converse_err, &target_model_id));
                            }
                        } else {
                            return Err(map_converse_error(converse_err, &target_model_id));
                        }
                    } else {
                        return Err(map_converse_error(converse_err, &target_model_id));
                    }
                }
                other_sdk_err => {
                    return Err(map_bedrock_sdk_error(other_sdk_err, &target_model_id));
                }
            },
        };

        let content = match response.output() {
            Some(ConverseOutput::Message(msg)) => {
                let mut text_parts = Vec::new();
                for block in msg.content() {
                    if let ContentBlock::Text(text) = block {
                        text_parts.push(text.as_str());
                    }
                }
                if text_parts.is_empty() {
                    None
                } else {
                    Some(text_parts.join(""))
                }
            }
            _ => None,
        };

        let finish_reason = match response.stop_reason() {
            StopReason::EndTurn | StopReason::StopSequence => FinishReason::Stop,
            StopReason::MaxTokens => FinishReason::Length,
            StopReason::ContentFiltered | StopReason::GuardrailIntervened => {
                FinishReason::ContentFilter
            }
            StopReason::ToolUse => FinishReason::ToolCalls,
            other => FinishReason::Other(other.as_str().to_owned()),
        };

        let usage = response
            .usage()
            .map(|u| TokenUsage {
                prompt_tokens: u32::try_from(u.input_tokens()).unwrap_or(u32::MAX),
                completion_tokens: u32::try_from(u.output_tokens()).unwrap_or(u32::MAX),
                total_tokens: u32::try_from(u.total_tokens()).unwrap_or(u32::MAX),
            })
            .unwrap_or_default();

        let model_name = primary_model_id.unwrap_or_else(|| target_model_id.clone());

        Ok(LlmResponse {
            content,
            tool_calls: Vec::new(),
            usage,
            finish_reason,
            refusal: None,
            model: model_name,
            reported_model,
        })
    }

    async fn list_models(&self) -> Result<Vec<ModelInfo>, LlmError> {
        let mut models = Vec::new();
        let mut seen_ids = HashSet::new();

        // 1. Any explicitly configured models on BedrockConfig
        for model_id in &self.config.models {
            if seen_ids.insert(model_id.clone()) {
                models.push(ModelInfo {
                    id: model_id.clone(),
                    description: Some("Configured Bedrock Model".to_owned()),
                });
            }
        }

        // 2. Any explicit inference profile on BedrockConfig
        if let Some(ref ip) = self.config.inference_profile {
            if seen_ids.insert(ip.clone()) {
                models.push(ModelInfo {
                    id: ip.clone(),
                    description: Some("Configured AWS Bedrock Inference Profile".to_owned()),
                });
            }
        }

        // 3. If control-plane discovery feature is enabled, query Bedrock Control Plane dynamically
        #[cfg(feature = "control-plane")]
        if let Some(cp_client) = self.control_plane_client().await {
            if let Ok(cp_models) = self.fetch_control_plane_models(cp_client).await {
                for m in cp_models {
                    if seen_ids.insert(m.id.clone()) {
                        models.push(m);
                    }
                }
            }
        }

        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bedrock_config_construction() {
        let config = BedrockConfig::new("us-west-2")
            .with_inference_profile("us.anthropic.claude-3-7-sonnet-20250219-v1:0")
            .with_endpoint_url("https://custom.bedrock.endpoint")
            .with_profile_name("custom-profile")
            .with_model("custom.model-v1")
            .with_models(["custom.model-v2", "custom.model-v3"]);
        assert_eq!(config.region, "us-west-2");
        assert_eq!(
            config.endpoint_url.as_deref(),
            Some("https://custom.bedrock.endpoint")
        );
        assert_eq!(config.profile_name.as_deref(), Some("custom-profile"));
        assert_eq!(
            config.inference_profile.as_deref(),
            Some("us.anthropic.claude-3-7-sonnet-20250219-v1:0")
        );
        assert_eq!(
            config.models,
            vec!["custom.model-v1", "custom.model-v2", "custom.model-v3"]
        );
    }

    #[test]
    fn test_bedrock_is_inference_profile() {
        assert!(is_inference_profile(
            "us.anthropic.claude-3-7-sonnet-20250219-v1:0"
        ));
        assert!(is_inference_profile(
            "eu.anthropic.claude-3-5-sonnet-20240620-v1:0"
        ));
        assert!(is_inference_profile(
            "apac.anthropic.claude-3-5-sonnet-20241022-v2:0"
        ));
        assert!(is_inference_profile(
            "cr.anthropic.claude-3-5-sonnet-20241022-v2:0"
        ));
        assert!(is_inference_profile(
            "global.anthropic.claude-3-7-sonnet-20250219-v1:0"
        ));
        assert!(is_inference_profile(
            "arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/my-profile"
        ));
        assert!(!is_inference_profile(
            "anthropic.claude-3-7-sonnet-20250219-v1:0"
        ));
        assert!(!is_inference_profile("amazon.nova-pro-v1:0"));
        assert!(!is_inference_profile("meta.llama3-3-70b-instruct-v1:0"));
    }

    #[test]
    fn test_bedrock_resolve_inference_profile_priority() {
        // 1. Explicit profile in model policy takes highest precedence
        let config = BedrockConfig::new("us-east-1").with_inference_profile("config-profile");
        let adapter =
            BedrockAdapter::new(config, BedrockCredentials::EnvironmentOrProfile).unwrap();

        let mut request = LlmRequest {
            model_policy: Default::default(),
            messages: vec![],
            tools: vec![],
            response_format: ResponseFormat::Text,
        };
        request.model_policy.profile = Some("request-profile".to_owned());

        let resolved =
            adapter.resolve_inference_profile(&request, Some("anthropic.claude-3-5-sonnet"));
        assert_eq!(resolved.as_deref(), Some("request-profile"));

        // 2. Global inference profile should NOT be prefixed with us.
        request.model_policy.profile = None;
        let resolved_global = adapter.resolve_inference_profile(
            &request,
            Some("global.anthropic.claude-3-7-sonnet-20250219-v1:0"),
        );
        assert_eq!(
            resolved_global.as_deref(),
            Some("global.anthropic.claude-3-7-sonnet-20250219-v1:0")
        );

        // 3. Application ARN inference profile should NOT be prefixed
        let resolved_arn = adapter.resolve_inference_profile(
            &request,
            Some("arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/custom-app"),
        );
        assert_eq!(
            resolved_arn.as_deref(),
            Some("arn:aws:bedrock:us-east-1:123456789012:application-inference-profile/custom-app")
        );

        // 4. Regional fallback for bare model without prefix
        let resolved_bare = adapter
            .resolve_inference_profile(&request, Some("anthropic.claude-3-5-sonnet-20241022-v2:0"));
        assert_eq!(
            resolved_bare.as_deref(),
            Some("us.anthropic.claude-3-5-sonnet-20241022-v2:0")
        );

        // 5. When no model is specified, fallback to config profile
        let resolved_no_model = adapter.resolve_inference_profile(&request, None);
        assert_eq!(resolved_no_model.as_deref(), Some("config-profile"));
    }

    #[test]
    fn test_bedrock_multi_model_routing_independence() {
        let config = BedrockConfig::new("us-east-1")
            .with_inference_profile("us.anthropic.claude-3-5-sonnet-20241022-v2:0");
        let adapter =
            BedrockAdapter::new(config, BedrockCredentials::EnvironmentOrProfile).unwrap();

        let request = LlmRequest {
            model_policy: Default::default(),
            messages: vec![],
            tools: vec![],
            response_format: ResponseFormat::Text,
        };

        // Routing Nova Pro does NOT get hijacked by the config's Claude 3.5 profile
        let nova = adapter.resolve_inference_profile(&request, Some("amazon.nova-pro-v1:0"));
        assert_eq!(nova.as_deref(), Some("us.amazon.nova-pro-v1:0"));

        // Routing Claude 3.7 global profile resolves cleanly
        let global_claude = adapter.resolve_inference_profile(
            &request,
            Some("global.anthropic.claude-3-7-sonnet-20250219-v1:0"),
        );
        assert_eq!(
            global_claude.as_deref(),
            Some("global.anthropic.claude-3-7-sonnet-20250219-v1:0")
        );

        // Routing Llama 3.3 regional profile resolves cleanly
        let llama =
            adapter.resolve_inference_profile(&request, Some("us.meta.llama3-3-70b-instruct-v1:0"));
        assert_eq!(llama.as_deref(), Some("us.meta.llama3-3-70b-instruct-v1:0"));
    }

    #[test]
    fn test_bedrock_credentials() {
        let creds = BedrockCredentials::Static {
            access_key_id: "AKIAIOSFODNN7EXAMPLE".to_owned(),
            secret_access_key: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY".to_owned(),
            session_token: Some("token123".to_owned()),
        };
        assert!(matches!(creds, BedrockCredentials::Static { .. }));

        let bearer = BedrockCredentials::BearerToken("secret-bearer-token".to_owned());
        assert_eq!(
            bearer,
            BedrockCredentials::BearerToken("secret-bearer-token".to_owned())
        );
    }

    #[tokio::test]
    async fn test_bedrock_adapter_bearer_token_client_init() {
        let adapter = BedrockAdapter::new(
            BedrockConfig::new("us-east-1"),
            BedrockCredentials::BearerToken("test-token-12345".to_owned()),
        )
        .unwrap();

        // Ensure client initialization with BearerToken succeeds
        let _client = adapter.client().await;
    }

    #[tokio::test]
    async fn test_bedrock_missing_model_and_inference_profile() {
        let adapter = BedrockAdapter::new(
            BedrockConfig::new("us-east-1"),
            BedrockCredentials::EnvironmentOrProfile,
        )
        .unwrap();

        let mut request = LlmRequest {
            model_policy: Default::default(),
            messages: vec![],
            tools: vec![],
            response_format: ResponseFormat::Text,
        };
        request.model_policy.profile = None;
        request.model_policy.model = None;

        let err = adapter.complete(request).await.unwrap_err();
        assert!(
            matches!(err, LlmError::Provider(msg) if msg.contains("model ID or inference profile must be specified"))
        );
    }

    #[tokio::test]
    async fn test_bedrock_adapter_list_models() {
        let adapter = BedrockAdapter::new(
            BedrockConfig::new("us-east-1")
                .with_model("custom.arbitrary-future-model")
                .with_models([
                    "global.anthropic.claude-5-sonnet",
                    "us.anthropic.claude-5-sonnet",
                ])
                .with_inference_profile("us.custom-application-profile"),
            BedrockCredentials::EnvironmentOrProfile,
        )
        .unwrap();

        let models = adapter.list_models().await.unwrap();
        assert!(!models.is_empty());

        // Configured custom models and profiles are recognized without any static gatekeeping
        assert!(
            models
                .iter()
                .any(|m| m.id == "custom.arbitrary-future-model")
        );
        assert!(
            models
                .iter()
                .any(|m| m.id == "global.anthropic.claude-5-sonnet")
        );
        assert!(
            models
                .iter()
                .any(|m| m.id == "us.anthropic.claude-5-sonnet")
        );
        assert!(
            models
                .iter()
                .any(|m| m.id == "us.custom-application-profile")
        );

        // Deduplication check: every model id in the list must be unique
        let mut seen = HashSet::new();
        for m in &models {
            assert!(seen.insert(&m.id), "Duplicate model id found: {}", m.id);
        }
    }

    #[tokio::test]
    async fn test_bedrock_unsupported_response_format() {
        let adapter = BedrockAdapter::new(
            BedrockConfig::new("us-east-1"),
            BedrockCredentials::EnvironmentOrProfile,
        )
        .unwrap();

        let request = LlmRequest {
            model_policy: Default::default(),
            messages: vec![],
            tools: vec![],
            response_format: ResponseFormat::JsonObject,
        };

        let err = adapter.complete(request).await.unwrap_err();
        assert!(matches!(err, LlmError::UnsupportedResponseFormat { .. }));
    }

    #[tokio::test]
    async fn test_bedrock_unsupported_tools() {
        use langchart_adapters::llm::ToolDefinition;
        let adapter = BedrockAdapter::new(
            BedrockConfig::new("us-east-1"),
            BedrockCredentials::EnvironmentOrProfile,
        )
        .unwrap();

        let request = LlmRequest {
            model_policy: Default::default(),
            messages: vec![],
            tools: vec![ToolDefinition {
                name: "test_tool".into(),
                description: "desc".into(),
                parameters: serde_json::json!({}),
            }],
            response_format: ResponseFormat::Text,
        };

        let err = adapter.complete(request).await.unwrap_err();
        assert!(matches!(err, LlmError::Provider(_)));
    }

    #[test]
    fn test_unhandled_converse_error() {
        let meta = aws_smithy_types::error::ErrorMetadata::builder()
            .code("CustomAWSException")
            .message("something broke in AWS")
            .build();
        let err = ConverseError::generic(meta);
        println!("display: {}", err);
        println!("debug: {:?}", err);
        println!("meta code: {:?}", err.meta().code());
        println!("meta message: {:?}", err.meta().message());

        let mapped = map_converse_error(err, "test-model");
        match mapped {
            LlmError::Provider(msg) => {
                assert!(msg.contains("CustomAWSException"));
                assert!(msg.contains("something broke in AWS"));
            }
            other => panic!("expected Provider error, got {other:?}"),
        }

        let io_err = std::io::Error::new(std::io::ErrorKind::TimedOut, "connection timed out");
        let err_unhandled = ConverseError::unhandled(io_err);
        let mapped_unhandled = map_converse_error(err_unhandled, "test-model");
        match mapped_unhandled {
            LlmError::Provider(msg) => {
                assert!(msg.contains("connection timed out"), "got msg: {msg}");
                assert!(!msg.ends_with(": unhandled error"));
            }
            other => panic!("expected Provider error, got {other:?}"),
        }
    }

    #[test]
    fn test_map_bedrock_sdk_error_variants() {
        use aws_smithy_runtime_api::client::result::SdkError;

        // 1. ConstructionFailure
        let construct_err: SdkError<ConverseError, HttpResponse> =
            SdkError::construction_failure("construction failed");
        let mapped = map_bedrock_sdk_error(construct_err, "test-model");
        assert!(
            matches!(mapped, LlmError::Provider(ref msg) if msg.contains("construction failed"))
        );

        // 2. TimeoutError
        let timeout_err: SdkError<ConverseError, HttpResponse> =
            SdkError::timeout_error("operation timed out after 180s");
        let mapped = map_bedrock_sdk_error(timeout_err, "test-model");
        match mapped {
            LlmError::Transport {
                stage,
                retryable,
                cause,
            } => {
                assert_eq!(stage, TransportStage::Send);
                assert!(retryable);
                assert!(cause.contains("timed out"), "got cause: {cause}");
            }
            other => panic!("expected Transport error, got {other:?}"),
        }

        // 3. DispatchFailure
        let connector_err = aws_smithy_runtime_api::client::result::ConnectorError::io(
            std::io::Error::new(
                std::io::ErrorKind::ConnectionReset,
                "connection reset by peer",
            )
            .into(),
        );
        let dispatch_err: SdkError<ConverseError, HttpResponse> =
            SdkError::dispatch_failure(connector_err);
        let mapped = map_bedrock_sdk_error(dispatch_err, "test-model");
        match mapped {
            LlmError::Transport {
                retryable, cause, ..
            } => {
                assert!(retryable);
                assert!(
                    cause.contains("connection reset by peer"),
                    "got cause: {cause}"
                );
            }
            other => panic!("expected Transport error, got {other:?}"),
        }

        // 4. ResponseError (HTTP 429)
        let http_res_429 = HttpResponse::new(
            aws_smithy_runtime_api::http::StatusCode::try_from(429).unwrap(),
            aws_smithy_types::body::SdkBody::empty(),
        );
        let resp_err_429: SdkError<ConverseError, HttpResponse> =
            SdkError::response_error("rate limit body", http_res_429);
        let mapped = map_bedrock_sdk_error(resp_err_429, "test-model");
        assert!(matches!(mapped, LlmError::RateLimited(ref msg) if msg.contains("429")));

        // 5. ResponseError (HTTP 504)
        let http_res_504 = HttpResponse::new(
            aws_smithy_runtime_api::http::StatusCode::try_from(504).unwrap(),
            aws_smithy_types::body::SdkBody::empty(),
        );
        let resp_err_504: SdkError<ConverseError, HttpResponse> =
            SdkError::response_error("gateway timeout", http_res_504);
        let mapped = map_bedrock_sdk_error(resp_err_504, "test-model");
        assert!(matches!(
            mapped,
            LlmError::Transport {
                stage: TransportStage::Headers,
                retryable: true,
                ..
            }
        ));
    }

    #[test]
    fn test_is_model_denied_or_inference_profile_required_meta() {
        let meta = aws_smithy_types::error::ErrorMetadata::builder()
            .code("AccessDeniedException")
            .message("You don't have access to this model")
            .build();
        let err = ConverseError::generic(meta);
        assert!(is_model_denied_or_inference_profile_required(&err));

        let meta2 = aws_smithy_types::error::ErrorMetadata::builder()
            .code("ValidationException")
            .message("Invocation through on-demand throughput isn't supported. Please use an inference profile.")
            .build();
        let err2 = ConverseError::generic(meta2);
        assert!(is_model_denied_or_inference_profile_required(&err2));
    }

    #[test]
    fn test_map_converse_modeled_variants() {
        use aws_sdk_bedrockruntime::types::error::{
            InternalServerException, ModelErrorException, ModelNotReadyException,
            ModelTimeoutException, ResourceNotFoundException, ServiceUnavailableException,
            ThrottlingException, ValidationException,
        };

        let err = ConverseError::ThrottlingException(
            ThrottlingException::builder()
                .message("too many requests")
                .build(),
        );
        assert!(matches!(
            map_converse_error(err, "m"),
            LlmError::RateLimited(_)
        ));

        let err = ConverseError::ModelNotReadyException(
            ModelNotReadyException::builder()
                .message("not ready")
                .build(),
        );
        assert!(matches!(
            map_converse_error(err, "m"),
            LlmError::ModelNotFound { .. }
        ));

        let err = ConverseError::ResourceNotFoundException(
            ResourceNotFoundException::builder()
                .message("not found")
                .build(),
        );
        assert!(matches!(
            map_converse_error(err, "m"),
            LlmError::ModelNotFound { .. }
        ));

        let err = ConverseError::ValidationException(
            ValidationException::builder()
                .message("max tokens exceeds context length")
                .build(),
        );
        assert!(matches!(
            map_converse_error(err, "m"),
            LlmError::ContextLengthExceeded
        ));

        let err = ConverseError::ModelTimeoutException(
            ModelTimeoutException::builder()
                .message("took too long")
                .build(),
        );
        assert!(
            matches!(map_converse_error(err, "m"), LlmError::Provider(ref msg) if msg.contains("timed out"))
        );

        let err = ConverseError::InternalServerException(
            InternalServerException::builder()
                .message("internal failure")
                .build(),
        );
        assert!(
            matches!(map_converse_error(err, "m"), LlmError::Provider(ref msg) if msg.contains("internal server error"))
        );

        let err = ConverseError::ServiceUnavailableException(
            ServiceUnavailableException::builder()
                .message("down for maintenance")
                .build(),
        );
        assert!(
            matches!(map_converse_error(err, "m"), LlmError::Provider(ref msg) if msg.contains("service unavailable"))
        );

        let err = ConverseError::ModelErrorException(
            ModelErrorException::builder()
                .message("model crashed")
                .build(),
        );
        assert!(
            matches!(map_converse_error(err, "m"), LlmError::Provider(ref msg) if msg.contains("model error"))
        );
    }
}
