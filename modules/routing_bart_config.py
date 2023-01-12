from transformers.models.bart.configuration_bart import BartConfig

class RoutingBartConfig(BartConfig):
    model_type = "bart"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=50265,
        max_position_embeddings=1024,
        encoder_layers=12,
        encoder_ffn_dim=4096,
        encoder_attention_heads=16,
        decoder_layers=12,
        decoder_ffn_dim=4096,
        decoder_attention_heads=16,
        encoder_layerdrop=0.0,
        decoder_layerdrop=0.0,
        activation_function="gelu",
        d_model=1024,
        dropout=0.1,
        attention_dropout=0.0,
        activation_dropout=0.0,
        init_std=0.02,
        classifier_dropout=0.0,
        scale_embedding=False,
        gradient_checkpointing=False,
        use_cache=True,
        num_labels=3,
        pad_token_id=1,
        bos_token_id=0,
        eos_token_id=2,
        is_encoder_decoder=True,
        decoder_start_token_id=2,
        forced_eos_token_id=2,
        router_input_dim=768,
        router_hidden_dim=64,
        router_block_num=3,
        router_rnn_layers=1,
        router_nn_type="rnn",
        router_pos_emb_concat=False,
        router_mode="softmax",
        encoder_vanilla_layers="",
        decoder_vanilla_layers="",
        exploration_epsilon=0.0,
        **kwargs
    ):
        super(BartConfig, self).__init__(
            num_labels=num_labels,
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            is_encoder_decoder=is_encoder_decoder,
            decoder_start_token_id=decoder_start_token_id,
            forced_eos_token_id=forced_eos_token_id,
            **kwargs,
        )

        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.d_model = d_model
        self.encoder_ffn_dim = encoder_ffn_dim
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.decoder_ffn_dim = decoder_ffn_dim
        self.decoder_layers = decoder_layers
        self.decoder_attention_heads = decoder_attention_heads
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.activation_function = activation_function
        self.init_std = init_std
        self.encoder_layerdrop = encoder_layerdrop
        self.decoder_layerdrop = decoder_layerdrop
        self.classifier_dropout = classifier_dropout
        self.use_cache = use_cache
        self.num_hidden_layers = encoder_layers
        self.gradient_checkpointing = gradient_checkpointing
        self.scale_embedding = scale_embedding  # scale factor will be sqrt(d_model) if True

        # Params for Routing Bart
        self.router_input_dim = router_input_dim
        self.router_hidden_dim = router_hidden_dim
        self.router_block_num = router_block_num
        self.router_mode = router_mode
        self.router_rnn_layers = router_rnn_layers
        self.router_pos_emb_concat = router_pos_emb_concat
        self.router_nn_type = router_nn_type
        self.encoder_vanilla_layers = encoder_vanilla_layers
        self.decoder_vanilla_layers = decoder_vanilla_layers
        self.exploration_epsilon = exploration_epsilon
        # End of params for Routing Bart

        # ensure backward compatibilty for BART CNN models
        if self.forced_bos_token_id is None and kwargs.get("force_bos_token_to_be_generated", False):
            self.forced_bos_token_id = self.bos_token_id
            warnings.warn(
                f"Please make sure the config includes `forced_bos_token_id={self.bos_token_id}` in future versions."
                "The config can simply be saved and uploaded again to be fixed."
            )

    @property
    def num_attention_heads(self) -> int:
        return self.encoder_attention_heads

    @property
    def hidden_size(self) -> int:
        return self.d_model