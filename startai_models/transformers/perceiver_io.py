# global
import startai
import startai_models
import string
import numpy as np

# local
from startai_models.transformers.helpers import (
    PreNorm,
    FeedForward,
    _perceiver_jax_weights_mapping,
)


# Specification class #
# --------------------#


class PerceiverIOSpec(startai.Container):
    def __init__(
        self,
        # input-output dependent
        input_dim,
        num_input_axes,
        output_dim,
        # input-output agnostic
        queries_dim=1024,
        network_depth=8,
        num_latents=512,
        latent_dim=1024,
        num_cross_att_heads=1,
        num_self_att_heads=8,
        cross_head_dim=261,
        latent_head_dim=128,
        weight_tie_layers=True,
        learn_query=True,
        query_shape=None,
        attn_dropout=0.0,
        fc_dropout=0.0,
        num_lat_att_per_layer=6,
        cross_attend_in_every_layer=False,
        with_decoder=True,
        with_final_head=True,
        fourier_encode_input=True,
        num_fourier_freq_bands=6,
        max_fourier_freq=None,
        device=None,
    ):
        if learn_query and not startai.exists(query_shape):
            raise Exception(
                "if learn_query is set, then query_shape must be specified."
            )

        device = startai.default(device, startai.default_device())

        super().__init__(
            input_dim=input_dim,
            num_input_axes=num_input_axes,
            output_dim=output_dim,
            queries_dim=queries_dim,
            network_depth=network_depth,
            num_latents=num_latents,
            latent_dim=latent_dim,
            num_cross_att_heads=num_cross_att_heads,
            num_self_att_heads=num_self_att_heads,
            cross_head_dim=cross_head_dim,
            latent_head_dim=latent_head_dim,
            weight_tie_layers=weight_tie_layers,
            learn_query=learn_query,
            query_shape=query_shape,
            attn_dropout=attn_dropout,
            fc_dropout=fc_dropout,
            num_lat_att_per_layer=num_lat_att_per_layer,
            cross_attend_in_every_layer=cross_attend_in_every_layer,
            with_decoder=with_decoder,
            with_final_head=with_final_head,
            fourier_encode_input=fourier_encode_input,
            num_fourier_freq_bands=num_fourier_freq_bands,
            max_fourier_freq=max_fourier_freq,
            device=device,
        )


# Main Class #
# -----------#


class PerceiverIO(startai.Module):
    def __init__(self, spec: PerceiverIOSpec, v: startai.Container = None, **kwargs):
        self._spec = spec
        startai.Module.__init__(self, v=v, **kwargs)

    def _create_latent_layer(self):
        return [
            [self._get_latent_attn(), self._get_fc()]
            for _ in range(self._spec.num_lat_att_per_layer)
        ]

    def _create_cross_layer(self):
        return {"cross_att": self._get_cross_attn(), "cross_fc": self._get_fc()}

    # noinspection PyUnusedLocal
    def _build(self, *args, **kwargs):
        self._fourier_encode_input = self._spec.fourier_encode_input
        fourier_channels = (
            (self._spec.num_input_axes * ((self._spec.num_fourier_freq_bands * 2) + 1))
            if self._spec.fourier_encode_input
            else 0
        )
        input_dim = fourier_channels + self._spec.input_dim

        self._get_cross_attn = lambda: PreNorm(
            self._spec.latent_dim,
            startai.MultiHeadAttention(
                self._spec.latent_dim,
                num_heads=self._spec.num_cross_att_heads,
                head_dim=self._spec.cross_head_dim,
                dropout_rate=self._spec.attn_dropout,
                key_dim=input_dim,
                value_dim=input_dim,
                device=self._spec.device,
            ),
            key_dim=input_dim,
            eps=1e-5,
            device=self._spec.device,
        )
        self._get_latent_attn = lambda: PreNorm(
            self._spec.latent_dim,
            startai.MultiHeadAttention(
                self._spec.latent_dim,
                num_heads=self._spec.num_self_att_heads,
                head_dim=self._spec.latent_head_dim,
                dropout_rate=self._spec.attn_dropout,
                device=self._spec.device,
            ),
            eps=1e-5,
            device=self._spec.device,
        )
        self._get_fc = lambda: PreNorm(
            self._spec.latent_dim,
            FeedForward(
                self._spec.latent_dim,
                dropout=self._spec.fc_dropout,
                device=self._spec.device,
            ),
            eps=1e-5,
            device=self._spec.device,
        )

        self._perceiver_encoder = list()
        if self._spec.weight_tie_layers:
            self._create_latent_layer = startai.cache_fn(self._create_latent_layer)
            self._create_cross_layer = startai.cache_fn(self._create_cross_layer)
        for i in range(self._spec.network_depth):
            layer = {"self_atts": self._create_latent_layer()}
            if i == 0 or self._spec.cross_attend_in_every_layer:
                layer = {**layer, **self._create_cross_layer()}
            self._perceiver_encoder.append(layer)

        self._classification_decoder = PreNorm(
            self._spec.queries_dim,
            startai.MultiHeadAttention(
                self._spec.queries_dim,
                num_heads=self._spec.num_cross_att_heads,
                head_dim=self._spec.latent_dim,
                key_dim=self._spec.latent_dim,
                value_dim=self._spec.latent_dim,
            ),
            key_dim=self._spec.latent_dim,
            eps=1e-5,
        )

        self._decoder = (
            PreNorm(
                self._spec.queries_dim,
                FeedForward(self._spec.queries_dim, device=self._spec.device),
                eps=1e-5,
            )
            if self._spec.with_decoder
            else None
        )

        self._decoder_logits = (
            startai.Linear(
                self._spec.queries_dim, self._spec.output_dim, device=self._spec.device
            )
            if self._spec.with_final_head
            else lambda x: x
        )

    def _create_variables(self, device, dtype=None):
        latents = startai.random_uniform(
            shape=(self._spec.num_latents, self._spec.latent_dim), device=device
        )
        # ToDo: set the correct initializatin scheme for the query here
        decoder_queries = (
            startai.random_uniform(shape=self._spec.query_shape + [self._spec.queries_dim])
            if self._spec.learn_query
            else None
        )
        return {"z_latents": latents, "decoder_queries": decoder_queries}

    def _forward(self, data, mask=None, queries=None):
        # shapes
        total_shape = data.shape
        batch_shape = total_shape[0 : -self._spec.num_input_axes - 1]
        data_shape = total_shape[-self._spec.num_input_axes - 1 : -1]

        # maybe flatten batch shape
        if batch_shape:
            num_batch_dims = len(batch_shape)
            batch_shape_keys = string.ascii_lowercase[0:num_batch_dims]
            batch_shape_str = " ".join(batch_shape_keys)
            batch_shape_dict = dict(zip(batch_shape_keys, batch_shape))
            flat_batch_size = int(np.prod(batch_shape))
            data = startai.einops_rearrange(
                data,
                "{} ... -> ({}) ...".format(batch_shape_str, batch_shape_str),
                **batch_shape_dict
            )
        else:
            flat_batch_size = 1
            data = startai.expand_dims(data, axis=0)

        # flatten the data channels
        data = startai.einops_rearrange(data, "b ... d -> b (...) d")

        # maybe add fourier positional encoding
        if self._fourier_encode_input:
            axis_pos = list(
                map(
                    lambda size: startai.linspace(
                        -1.0, 1.0, num=size, device=self._spec.device
                    ),
                    data_shape,
                )
            )
            pos = startai.stack(startai.meshgrid(*axis_pos), axis=-1)
            pos_flat = startai.reshape(pos, [-1, len(axis_pos)])
            if not startai.exists(self._spec.max_fourier_freq):
                self._spec.max_fourier_freq = startai.array(data_shape, dtype="float32")
            enc_pos = startai.fourier_encode(
                pos_flat,
                self._spec.max_fourier_freq,
                num_bands=self._spec.num_fourier_freq_bands,
                linear=True,
                flatten=True,
            )
            enc_pos = startai.einops_repeat(enc_pos, "... -> b ...", b=flat_batch_size)
            data = startai.concat([data, enc_pos], axis=-1)

        # batchify latents
        x = startai.einops_repeat(self.v.z_latents, "n d -> b n d", b=flat_batch_size)

        # layers
        for layer_dict in self._perceiver_encoder:
            if "cross_att" in layer_dict:
                x = layer_dict["cross_att"](x, data, data, attention_mask=mask) + x
            if "cross_fc" in layer_dict:
                x = layer_dict["cross_fc"](x) + x

            for self_attn, self_fc in layer_dict["self_atts"]:
                x = self_attn(x) + x
                x = self_fc(x) + x

        # queries
        if not startai.exists(queries):
            if startai.exists(self.v.decoder_queries):
                queries = startai.einops_repeat(
                    self.v.decoder_queries, "... -> b ...", b=flat_batch_size
                )
            else:
                raise Exception(
                    "If learn_query is not set as True, the queries must "
                    "be provided explicitly during the forward pass."
                )

        queries_shape = list(queries.shape)

        queries = startai.einops_rearrange(queries, "b ... d -> b (...) d")

        # cross attend from decoder queries to latents

        latents = self._classification_decoder(queries, x, x)

        # optional decoder feedforward

        if startai.exists(self._decoder):
            latents = latents + self._decoder(latents)

        # final linear out

        ret_flat = self._decoder_logits(latents)

        # reshape to correct number of axes
        ret_flat = startai.reshape(ret_flat, queries_shape[:-1] + [self._spec.output_dim])

        # return with original batch shape
        if batch_shape:
            return startai.einops_rearrange(
                ret_flat,
                "({}) ... -> {} ...".format(batch_shape_str, batch_shape_str),
                **batch_shape_dict
            )
        return ret_flat[0]


def perceiver_io_img_classification(spec, pretrained=True):
    if not pretrained:
        return PerceiverIO(spec)

    reference_model = PerceiverIO(spec)
    url = "https://storage.googleapis.com/perceiver_io/imagenet_learned_position_encoding.pystate"  # noqa
    w_clean = startai_models.helpers.load_jax_weights(
        url,
        reference_model,
        custom_mapping=_perceiver_jax_weights_mapping,
        special_rename={"mlp": "net"},
        with_mha=True,
    )
    reference_model.v = w_clean
    return reference_model
